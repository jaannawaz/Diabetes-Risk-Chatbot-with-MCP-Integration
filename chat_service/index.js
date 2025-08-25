const express = require('express');
const cors = require('cors');
const axios = require('axios');
const dotenv = require('dotenv');
const { z } = require('zod');
const multer = require('multer');
const Papa = require('papaparse');
const puppeteer = require('puppeteer');
const handlebars = require('handlebars');
const fs = require('fs');
const path = require('path');

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());
const upload = multer({ storage: multer.memoryStorage() });

const MODEL_URL = process.env.MODEL_URL || 'http://127.0.0.1:8001';
const MCP_URL = process.env.MCP_URL || 'http://127.0.0.1:8002';
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL || 'https://api.openai.com/v1';
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-4o-mini';
const GROK_API_KEY = process.env.GROK_API_KEY || '';
const GROK_BASE_URL = process.env.GROK_BASE_URL || 'https://api.x.ai/v1';
const GROK_MODEL = process.env.GROK_MODEL || 'grok-2-mini';

const safetyDisclaimerEN = 'This assistant is not a medical diagnosis. For urgent symptoms, seek emergency care.';
const safetyDisclaimerAR = 'هذا المساعد ليس تشخيصًا طبيًا. إذا كانت لديك أعراض مقلقة، يُرجى طلب الرعاية الطارئة.';

const ChatInput = z.object({
  message: z.string(),
  language: z.enum(['auto', 'en', 'ar']).default('auto'),
  features: z.object({
    gender: z.enum(['Male', 'Female']),
    age: z.number(),
    hypertension: z.number().int().min(0).max(1),
    heart_disease: z.number().int().min(0).max(1),
    smoking_history: z.string(),
    bmi: z.number(),
    HbA1c_level: z.number(),
    blood_glucose_level: z.number(),
  }),
  patient_id: z.string().optional(),
  variant_id: z.string().optional(),
});

function localize(languageHint, en, ar) {
  if (languageHint === 'ar') return ar;
  if (languageHint === 'en') return en;
  return en; // auto → default EN for demo
}

function formatResponse(lang, risk, label, topFactors, extras) {
  const en = {
    text: `Estimated diabetes risk: ${(risk*100).toFixed(1)}% (label: ${label}). Top factors: ${topFactors.join(', ')}.`,
    disclaimer: safetyDisclaimerEN,
    ...extras,
  };
  const ar = {
    text: `احتمال خطر السكري: ${(risk*100).toFixed(1)}% (التصنيف: ${label}). العوامل الأهم: ${topFactors.join(', ')}.`,
    disclaimer: safetyDisclaimerAR,
    ...extras,
  };
  return localize(lang, en, ar);
}

async function maybePolishWithLLM(language, text) {
  const prompt = language === 'ar'
    ? 'أعد صياغة النص بشكل واضح ومهني باللغة العربية دون تغيير الأرقام أو النسب.'
    : 'Rewrite the text clearly and professionally in English. Do not change numbers or percentages.';

  // Prefer OpenAI if configured
  if (OPENAI_API_KEY) {
    try {
      const resp = await axios.post(
        `${OPENAI_BASE_URL}/chat/completions`,
        {
          model: OPENAI_MODEL,
          messages: [
            { role: 'system', content: prompt },
            { role: 'user', content: text },
          ],
          temperature: 0.2,
        },
        {
          headers: {
            'Authorization': `Bearer ${OPENAI_API_KEY}`,
            'Content-Type': 'application/json',
          },
          timeout: 4000,
        }
      );
      const choice = resp.data?.choices?.[0];
      const content = choice?.message?.content || choice?.text;
      return content || text;
    } catch (_) {
      // fall through to GROK if available
    }
  }

  if (GROK_API_KEY) {
    try {
      const resp = await axios.post(
        `${GROK_BASE_URL}/chat/completions`,
        {
          model: GROK_MODEL,
          messages: [
            { role: 'system', content: prompt },
            { role: 'user', content: text },
          ],
          temperature: 0.2,
        },
        {
          headers: {
            'Authorization': `Bearer ${GROK_API_KEY}`,
            'Content-Type': 'application/json',
          },
          timeout: 4000,
        }
      );
      const choice = resp.data?.choices?.[0];
      const content = choice?.message?.content || choice?.text;
      return content || text;
    } catch (_) {}
  }

  return text;
}

app.get('/health', (_req, res) => {
  res.json({ status: 'ok' });
});

app.post('/chat', async (req, res) => {
  const parse = ChatInput.safeParse(req.body);
  if (!parse.success) {
    return res.status(400).json({ error: 'Invalid payload', details: parse.error.flatten() });
  }
  const { message, language, features, patient_id, variant_id } = parse.data;
  try {
    const pred = await axios.post(`${MODEL_URL}/predict`, features, { timeout: 2000 });
    const { risk_score, risk_label, top_factors } = pred.data;

    let labs = null;
    if (patient_id) {
      try {
        const r = await axios.post(`${MCP_URL}/tools/labs.getLatestHbA1c`, { patient_id }, { headers: { 'X-Client-Id': 'chat' }, timeout: 1500 });
        labs = r.data;
      } catch {}
    }

    let guideline = null;
    try {
      const g = await axios.get(`${MCP_URL}/tools/guidelines.lookup`, { params: { topic: 'hba1c' }, timeout: 1500 });
      guideline = g.data;
    } catch {}

    // Global explain visuals (for clinician mode)
    let explain = null;
    try {
      const eg = await axios.get(`${MODEL_URL}/explain/global`, { timeout: 1500 });
      explain = eg.data; // contains paths under /assets
    } catch {}

    // Genomics (optional)
    let genomics = null;
    if (variant_id) {
      try {
        const gv = await axios.get(`${MCP_URL}/tools/genomics.lookup`, { params: { variant_id }, timeout: 1500 });
        genomics = gv.data;
      } catch {}
    }

    const extras = {
      risk_score,
      risk_label,
      top_factors,
      labs,
      guideline,
      explain,
      genomics,
    };
    // Basic sentiment detection (naive)
    const txt = (message || '').toLowerCase();
    const negWords = ['worried','anxious','scared','pain','tired','weak','bad','fear'];
    const posWords = ['ok','fine','good','better','improve','calm'];
    let sentiment = 'neutral';
    if (negWords.some(w => txt.includes(w))) sentiment = 'negative';
    else if (posWords.some(w => txt.includes(w))) sentiment = 'positive';

    // Category by risk and labs
    const hba1c = Number(features.HbA1c_level);
    const glucose = Number(features.blood_glucose_level);
    let category = 'healthy';
    if (risk_label === 'high' || hba1c >= 6.5 || glucose >= 126) category = 'diabetic';
    else if (risk_label === 'medium' || (hba1c >= 5.7 && hba1c < 6.5) || (glucose >= 100 && glucose < 126)) category = 'prediabetic';

    const tips = {
      en: {
        diabetic: {
          diet: [
            'Focus on whole foods: vegetables, lean proteins, legumes.',
            'Choose high-fiber carbs (whole grains); avoid sugary drinks.',
            'Distribute carbs evenly across meals; monitor portions.',
          ],
          workout: [
            '150+ minutes/week moderate aerobic (e.g., brisk walking).',
            '2–3 sessions/week resistance training (major muscle groups).',
            'Break up long sitting with short walks.',
          ],
          sleep: [
            'Aim for 7–9 hours/night, consistent schedule.',
            'Limit screens 1 hour before bed; reduce caffeine after noon.',
          ],
        },
        prediabetic: {
          diet: [
            'Increase vegetables and fiber; choose low-GI carbs.',
            'Reduce refined sugars; hydrate with water.',
            'Track weight trend; small calorie deficit may help.',
          ],
          workout: [
            '≥150 minutes/week aerobic activity; add light resistance work.',
            'Incorporate daily steps goals (e.g., 7–10k).',
          ],
          sleep: [
            'Keep a regular bedtime; target 7–9 hours.',
            'Practice wind-down routine (dim lights, reading).',
          ],
        },
        healthy: {
          diet: [
            'Maintain balanced plate: half vegetables, lean protein, whole grains.',
            'Limit ultra-processed foods and sugary beverages.',
          ],
          workout: [
            'Stay active most days; mix cardio and strength work.',
          ],
          sleep: [
            'Keep consistent schedule; 7–9 hours ideal.',
          ],
        },
      },
      ar: {
        diabetic: {
          diet: [
            'التركيز على الأطعمة الطبيعية: الخضار، البروتينات الخالية من الدهون، البقوليات.',
            'اختيار كربوهيدرات غنية بالألياف وتجنب المشروبات السكرية.',
            'توزيع الكربوهيدرات على الوجبات مع ضبط الحصص.',
          ],
          workout: [
            '150+ دقيقة أسبوعياً من نشاط هوائي معتدل (مثل المشي السريع).',
            'تمارين مقاومة 2–3 مرات أسبوعياً.',
            'تفادي الجلوس لفترات طويلة مع فواصل حركة قصيرة.',
          ],
          sleep: [
            'النوم 7–9 ساعات بجدول ثابت.',
            'تقليل الشاشات قبل النوم والحد من الكافيين بعد الظهر.',
          ],
        },
        prediabetic: {
          diet: [
            'زيادة الخضار والألياف واختيار كربوهيدرات منخفضة المؤشر الجلايسيمي.',
            'تقليل السكريات المكررة وزيادة شرب الماء.',
            'متابعة الوزن مع عجز حراري بسيط عند الحاجة.',
          ],
          workout: [
            '≥150 دقيقة/أسبوع نشاط هوائي مع تمارين مقاومة خفيفة.',
            'زيادة عدد الخطوات اليومية (مثلاً 7–10 آلاف).',
          ],
          sleep: [
            'نظام نوم منتظم؛ استهداف 7–9 ساعات.',
            'روتين استرخاء قبل النوم.',
          ],
        },
        healthy: {
          diet: [
            'وجبة متوازنة: نصفها خضار مع بروتينات وكربوهيدرات كاملة.',
            'تقليل الأطعمة فائقة التصنيع والمشروبات السكرية.',
          ],
          workout: [
            'نشاط بدني معظم الأيام مع تنويع التمارين.',
          ],
          sleep: [
            'النوم 7–9 ساعات بانتظام.',
          ],
        },
      },
    };

    const langKey = language === 'ar' ? 'ar' : 'en';
    const rec = tips[langKey][category];
    const followup = langKey === 'ar'
      ? 'اقتراح: إعادة التقييم خلال 1–2 أسبوع، وطلب فحص HbA1c/الجلوكوز إذا لزم.'
      : 'Suggestion: re-check in 1–2 weeks and consider HbA1c/glucose testing as needed.';

    extras.recommendations = { category, sentiment, ...rec, followup };

    const out = formatResponse(language, risk_score, risk_label, top_factors, extras);
    out.text = await maybePolishWithLLM(language, out.text);
    res.json(out);
  } catch (e) {
    res.status(500).json({ error: 'Failed to compose response', detail: String(e) });
  }
});

const port = process.env.PORT || 8003;
if (require.main === module) {
  app.listen(port, () => {
    console.log(`Chat service listening on :${port}`);
  });
}

module.exports = app;

// Labs upload parsing (CSV only, mock)
app.post('/upload/labs', upload.single('file'), (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
    const text = req.file.buffer.toString('utf-8');
    const parsed = Papa.parse(text, { header: true });
    if (parsed.errors?.length) {
      return res.status(400).json({ error: 'CSV parse error', details: parsed.errors.slice(0,3) });
    }
    // Simple heuristics to extract HbA1c and glucose
    let hba1c = null, glucose = null;
    for (const row of parsed.data) {
      const keys = Object.keys(row).map(k => k.toLowerCase());
      const vals = Object.values(row);
      for (let i=0;i<keys.length;i++) {
        const k = keys[i]; const v = vals[i];
        if (hba1c == null && /hba1c|a1c/.test(k)) {
          const num = parseFloat(String(v).replace(/[^0-9.]/g,''));
          if (!Number.isNaN(num)) hba1c = num;
        }
        if (glucose == null && /glucose|fpg|fasting/.test(k)) {
          const num = parseFloat(String(v).replace(/[^0-9.]/g,''));
          if (!Number.isNaN(num)) glucose = num;
        }
      }
    }
    return res.json({ HbA1c_level: hba1c, blood_glucose_level: glucose });
  } catch (e) {
    return res.status(500).json({ error: 'Upload parse failed', detail: String(e) });
  }
});

// Wearables stub (steps/hr/activity) — returns normalized modifiers
app.post('/wearables/ingest', express.json(), (req, res) => {
  const { steps = 6000, heart_rate = 78, activity_min = 30 } = req.body || {};
  const activity_score = Math.min(1, (Number(activity_min) || 0) / 60);
  const steps_score = Math.min(1, (Number(steps) || 0) / 10000);
  const hr_score = Math.max(0, Math.min(1, (100 - (Number(heart_rate) || 0)) / 50));
  return res.json({ modifiers: { activity_score, steps_score, hr_score } });
});

// Voice-to-text (optional, OpenAI Whisper) — expects audio file
app.post('/voice/transcribe', upload.single('audio'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No audio uploaded' });
    const OPENAI_API_KEY = process.env.OPENAI_API_KEY || '';
    if (!OPENAI_API_KEY) return res.status(400).json({ error: 'OPENAI_API_KEY not configured' });
    // Minimal proxy to OpenAI transcription API
    const form = new (require('form-data'))();
    form.append('file', req.file.buffer, { filename: 'audio.webm', contentType: req.file.mimetype || 'audio/webm' });
    form.append('model', 'whisper-1');
    const r = await axios.post('https://api.openai.com/v1/audio/transcriptions', form, {
      headers: { ...form.getHeaders(), Authorization: `Bearer ${OPENAI_API_KEY}` },
      timeout: 15000,
    });
    return res.json({ text: r.data.text || '' });
  } catch (e) {
    return res.status(500).json({ error: 'Transcription failed', detail: String(e) });
  }
});

// Generate PDF Report
app.post('/generate-report', express.json(), async (req, res) => {
  try {
    const { userData, riskData, factors, recommendations, labResults, language = 'en' } = req.body;
    
    if (!userData || !riskData) {
      return res.status(400).json({ error: 'Missing required data for report generation' });
    }

    const templatePath = path.join(__dirname, 'templates', 'report.hbs');
    const templateSource = fs.readFileSync(templatePath, 'utf8');
    const template = handlebars.compile(templateSource);

    // Localized strings
    const strings = {
      en: {
        title: 'Diabetes Risk Assessment Report',
        subtitle: 'Personalized Health Analysis',
        riskAssessment: { title: 'Risk Assessment', risk: 'Risk' },
        userInfo: { age: 'Age' },
        keyFactors: { title: 'Key Risk Factors' },
        labResults: { title: 'Laboratory Results' },
        recommendations: { title: 'Recommendations' },
        disclaimer: { 
          title: 'Medical Disclaimer',
          text: 'This report is for informational purposes only and should not replace professional medical advice. Please consult your healthcare provider for proper diagnosis and treatment.'
        },
        footer: {
          generatedBy: 'Generated by Diabetes Risk Assessment System',
          timestamp: `Generated on ${new Date().toLocaleString('en-US')}`
        }
      },
      ar: {
        title: 'تقرير تقييم مخاطر السكري',
        subtitle: 'تحليل صحي شخصي',
        riskAssessment: { title: 'تقييم المخاطر', risk: 'المخاطر' },
        userInfo: { age: 'العمر' },
        keyFactors: { title: 'عوامل الخطر الرئيسية' },
        labResults: { title: 'نتائج المختبر' },
        recommendations: { title: 'التوصيات' },
        disclaimer: { 
          title: 'إخلاء مسؤولية طبية',
          text: 'هذا التقرير لأغراض إعلامية فقط ولا يجب أن يحل محل المشورة الطبية المهنية. يرجى استشارة مقدم الرعاية الصحية للحصول على التشخيص والعلاج المناسب.'
        },
        footer: {
          generatedBy: 'تم إنشاؤه بواسطة نظام تقييم مخاطر السكري',
          timestamp: `تم الإنشاء في ${new Date().toLocaleString('ar-EG')}`
        }
      }
    };

    const isRTL = language === 'ar';
    const langStrings = strings[language] || strings.en;

    const templateData = {
      lang: language,
      direction: isRTL ? 'rtl' : 'ltr',
      header: {
        title: langStrings.title,
        subtitle: langStrings.subtitle
      },
      sections: langStrings,
      userData: {
        age: userData.age,
        bmi: userData.bmi?.toFixed(1) || 'N/A'
      },
      riskData: {
        score: (riskData.risk_score * 100).toFixed(1),
        label: riskData.risk_label
      },
      factors: factors?.map(f => ({
        name: f.name,
        value: f.value,
        impact: f.impact > 0 ? 'increases risk' : 'decreases risk'
      })) || [],
      recommendations: recommendations || [],
      labResults: labResults?.map(lab => ({
        name: lab.name,
        value: lab.value,
        unit: lab.unit || ''
      })) || null,
      footer: langStrings.footer
    };

    const html = template(templateData);

    // Launch Puppeteer and generate PDF
    const browser = await puppeteer.launch({ 
      headless: 'new',
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    const page = await browser.newPage();
    await page.setContent(html, { waitUntil: 'networkidle0' });
    
    const pdf = await page.pdf({
      format: 'A4',
      printBackground: true,
      margin: { top: '1cm', right: '1cm', bottom: '1cm', left: '1cm' }
    });

    await browser.close();

    res.setHeader('Content-Type', 'application/pdf');
    res.setHeader('Content-Disposition', `attachment; filename="diabetes-risk-report-${Date.now()}.pdf"`);
    res.send(pdf);

  } catch (error) {
    console.error('Report generation error:', error);
    res.status(500).json({ error: 'Failed to generate report', detail: error.message });
  }
});


