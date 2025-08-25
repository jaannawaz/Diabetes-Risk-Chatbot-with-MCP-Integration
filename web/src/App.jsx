import { useMemo, useState } from 'react'
import './App.css'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
} from 'chart.js'
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend)

const CHAT_API = import.meta.env.VITE_CHAT_API_URL || 'http://127.0.0.1:8003'

function App() {
  const [language, setLanguage] = useState('en')
  const [message, setMessage] = useState('')
  const [features, setFeatures] = useState({
    gender: 'Male',
    age: 45,
    hypertension: 0,
    heart_disease: 0,
    smoking_history: 'never',
    bmi: 27.5,
    HbA1c_level: 6.0,
    blood_glucose_level: 145,
  })
  const [patientId, setPatientId] = useState('P001')
  const [response, setResponse] = useState(null)
  const [messages, setMessages] = useState([])
  const [history, setHistory] = useState([])
  const [consented, setConsented] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [wearables, setWearables] = useState({ steps: 6000, heart_rate: 78, activity_min: 30 })
  const [bmiCalc, setBmiCalc] = useState({ height_cm: 170, weight_kg: 75, bmi: 25.9 })
  const [variantId, setVariantId] = useState('')
  const [transcribing, setTranscribing] = useState(false)
  const [reportLoading, setReportLoading] = useState(false)
  const [lastAssessment, setLastAssessment] = useState(null)

  const isRTL = useMemo(() => language === 'ar', [language])

  async function send() {
    if (!message.trim()) return
    const userMsg = { role: 'user', text: message }
    setMessages((m) => [...m, userMsg])
    setMessage('')
    const payload = { message: userMsg.text, language, features, patient_id: patientId, variant_id: variantId || undefined }
    try {
      const res = await fetch(`${CHAT_API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await res.json()
      setResponse(data)
      if (typeof data?.risk_score === 'number') {
        setHistory(h => [...h, { t: new Date().toLocaleTimeString(), v: data.risk_score }].slice(-20))
        // Store last assessment for report generation
        setLastAssessment({
          userData: { age: features.age, bmi: features.bmi },
          riskData: { risk_score: data.risk_score, risk_label: data.risk_label },
          factors: data.factors || [],
          recommendations: data.recommendations || [],
          labResults: data.lab_results || null,
          language: language
        })
      }
      const botMsg = { role: 'assistant', text: data.text, meta: data }
      setMessages((m) => [...m, botMsg])
    } catch (e) {
      const errMsg = { role: 'assistant', text: (isRTL ? 'تعذر إتمام الطلب.' : 'Failed to process request.') }
      setMessages((m) => [...m, errMsg])
    }
  }

  async function generateReport() {
    if (!lastAssessment) {
      alert(isRTL ? 'لا يوجد تقييم للمخاطر لإنشاء التقرير' : 'No risk assessment available to generate report')
      return
    }
    
    setReportLoading(true)
    try {
      const res = await fetch(`${CHAT_API}/generate-report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(lastAssessment),
      })
      
      if (res.ok) {
        const blob = await res.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `diabetes-risk-report-${Date.now()}.pdf`
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
      } else {
        throw new Error('Failed to generate report')
      }
    } catch (e) {
      alert(isRTL ? 'فشل في إنشاء التقرير' : 'Failed to generate report')
    }
    setReportLoading(false)
  }

  return (
    <div dir={isRTL ? 'rtl' : 'ltr'} className="layout">
      {!consented && (
        <div className="modal">
          <div className="modal-content">
            <h3>{isRTL ? 'الموافقة' : 'Consent'}</h3>
            <p>{isRTL ? 'هذا التطبيق تجريبي وليس لتشخيص طبي.' : 'This app is a demo and not for medical diagnosis.'}</p>
            <button onClick={() => setConsented(true)}>{isRTL ? 'أوافق' : 'I Agree'}</button>
          </div>
        </div>
      )}

      <header className="topbar">
        <h2 className="brand">{isRTL ? 'مساعد خطر السكري' : 'Diabetes Risk Assistant'}</h2>
        <div className="spacer" />
        <select value={language} onChange={(e) => setLanguage(e.target.value)}>
          <option value="en">EN</option>
          <option value="ar">AR</option>
        </select>
      </header>

      <div className="twoPanel">
        <aside className="leftPane">
          <div className="card" style={{marginBottom:'0.75rem'}}>
            <div style={{fontWeight:600, marginBottom:'0.4rem'}}>{isRTL ? 'المدخلات' : 'User Inputs'}</div>
            <div className="grid">
              <label>Gender
                <select value={features.gender} onChange={(e) => setFeatures(f => ({...f, gender: e.target.value}))}>
                  <option>Male</option>
                  <option>Female</option>
                </select>
              </label>
              <label>Age
                <input type="number" value={features.age} onChange={(e) => setFeatures(f => ({...f, age: Number(e.target.value)}))} />
              </label>
              <label>Hypertension
                <select value={features.hypertension} onChange={(e) => setFeatures(f => ({...f, hypertension: Number(e.target.value)}))}>
                  <option value={0}>0</option>
                  <option value={1}>1</option>
                </select>
              </label>
              <label>Heart Disease
                <select value={features.heart_disease} onChange={(e) => setFeatures(f => ({...f, heart_disease: Number(e.target.value)}))}>
                  <option value={0}>0</option>
                  <option value={1}>1</option>
                </select>
              </label>
              <label>Smoking
                <input value={features.smoking_history} onChange={(e) => setFeatures(f => ({...f, smoking_history: e.target.value}))} />
              </label>
              <label>BMI
                <input type="number" value={features.bmi} step="0.1" onChange={(e) => setFeatures(f => ({...f, bmi: Number(e.target.value)}))} />
              </label>
              <label>HbA1c
                <input type="number" value={features.HbA1c_level} step="0.1" onChange={(e) => setFeatures(f => ({...f, HbA1c_level: Number(e.target.value)}))} />
              </label>
              <label>Glucose
                <input type="number" value={features.blood_glucose_level} onChange={(e) => setFeatures(f => ({...f, blood_glucose_level: Number(e.target.value)}))} />
              </label>
              <label>Patient ID
                <input value={patientId} onChange={(e) => setPatientId(e.target.value)} />
              </label>
              <label>Variant ID
                <input value={variantId} onChange={(e) => setVariantId(e.target.value)} placeholder="e.g., TCF7L2:rs7903146" />
              </label>
            </div>
          </div>

          <div className="card" style={{marginBottom:'0.75rem'}}>
            <div style={{fontWeight:600, marginBottom:'0.4rem'}}>{isRTL ? 'المختبرات' : 'Labs'}</div>
            <label>{isRTL ? 'رفع نتائج المختبر (CSV)' : 'Upload Lab Results (CSV)'}
              <input type="file" accept=".csv" onChange={async (e)=>{
                const f = e.target.files?.[0]; if(!f) return;
                setUploading(true);
                const fd = new FormData(); fd.append('file', f);
                try {
                  const res = await fetch(`${CHAT_API}/upload/labs`, { method: 'POST', body: fd });
                  const data = await res.json();
                  setFeatures(prev => ({...prev,
                    HbA1c_level: data.HbA1c_level ?? prev.HbA1c_level,
                    blood_glucose_level: data.blood_glucose_level ?? prev.blood_glucose_level,
                  }));
                } catch {}
                setUploading(false);
              }} />
            </label>
            {uploading && <div style={{fontSize:'0.8rem', opacity:0.8}}>{isRTL ? 'جارٍ التحميل...' : 'Uploading...'}</div>}
          </div>

          <div className="card" style={{marginBottom:'0.75rem'}}>
            <div style={{fontWeight:600, marginBottom:'0.4rem'}}>{isRTL ? 'الأجهزة القابلة للارتداء' : 'Wearables'}</div>
            <div className="grid">
              <label>Steps
                <input type="number" value={wearables.steps} onChange={(e)=> setWearables(w=>({...w, steps: Number(e.target.value)}))} />
              </label>
              <label>Heart Rate
                <input type="number" value={wearables.heart_rate} onChange={(e)=> setWearables(w=>({...w, heart_rate: Number(e.target.value)}))} />
              </label>
              <label>Activity (min)
                <input type="number" value={wearables.activity_min} onChange={(e)=> setWearables(w=>({...w, activity_min: Number(e.target.value)}))} />
              </label>
            </div>
            <div style={{display:'flex', gap:'0.5rem', marginTop:'0.5rem'}}>
              <button onClick={async ()=>{
                try{
                  const r = await fetch(`${CHAT_API}/wearables/ingest`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(wearables)});
                  const data = await r.json();
                  if(data.modifiers){ setFeatures(f=>({...f, bmi: Math.max(10, f.bmi * (1 - 0.02*data.modifiers.activity_score))})) }
                }catch{}
              }}>{isRTL ? 'تطبيق' : 'Apply'}</button>
            </div>
          </div>

          <div className="card" style={{marginBottom:'0.75rem'}}>
            <div style={{fontWeight:600, marginBottom:'0.4rem'}}>{isRTL? 'حاسبة مؤشر كتلة الجسم' : 'BMI Calculator'}</div>
            <div className="grid">
              <label>{isRTL? 'الطول (سم)' : 'Height (cm)'}
                <input type="number" value={bmiCalc.height_cm} onChange={(e)=> setBmiCalc(s=> ({...s, height_cm: Number(e.target.value)}))} />
              </label>
              <label>{isRTL? 'الوزن (كجم)' : 'Weight (kg)'}
                <input type="number" value={bmiCalc.weight_kg} onChange={(e)=> setBmiCalc(s=> ({...s, weight_kg: Number(e.target.value)}))} />
              </label>
            </div>
            <div style={{display:'flex', gap:'0.5rem', alignItems:'center', marginTop:'0.5rem'}}>
              <button onClick={()=>{
                const h = bmiCalc.height_cm/100; const w = bmiCalc.weight_kg; if(h>0){ const b = w/(h*h); setBmiCalc(s=>({...s, bmi: Math.round(b*10)/10})); }
              }}>{isRTL? 'حساب' : 'Calculate'}</button>
              <div>{isRTL? 'BMI:' : 'BMI:'} {bmiCalc.bmi}</div>
              <button onClick={()=> setFeatures(f=> ({...f, bmi: bmiCalc.bmi}))}>{isRTL? 'تطبيق على الميزات' : 'Apply to Features'}</button>
            </div>
      </div>

          <div className="disclaimer">{isRTL ? 'هذا التطبيق لأغراض توضيحية فقط.' : 'For demonstration purposes only.'}</div>
        </aside>

        <main className="rightPane">
          <div className="chatWindow">
            {messages.length === 0 && (
              <div className="empty">{isRTL ? 'ابدأ المحادثة بإرسال رسالة.' : 'Start the conversation by sending a message.'}</div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`bubble ${m.role}`}>
                <div className="text">{m.text}</div>
                {m.role === 'assistant' && m.meta && (
      <div className="card">
                    {typeof m.meta.risk_score === 'number' && (
                      <div className="riskHeader">
                        <span className={`badge ${m.meta.risk_label}`}>{m.meta.risk_label}</span>
                        <div className="progress"><div className="progressFill" style={{ width: `${Math.min(100, Math.max(0, m.meta.risk_score*100))}%` }}></div></div>
                        <span className="pct">{(m.meta.risk_score*100).toFixed(1)}%</span>
                      </div>
                    )}
                    <div style={{display:'flex', gap:'0.75rem', alignItems:'center'}}>
                      <span style={{fontSize:'1.2rem'}}>🩺</span>
                      <div><strong>{isRTL ? 'الخطر' : 'Risk'}:</strong> {(m.meta.risk_score * 100).toFixed(1)}% ({m.meta.risk_label})</div>
                    </div>
                    <div style={{display:'flex', gap:'0.75rem', alignItems:'center'}}>
                      <span style={{fontSize:'1.2rem'}}>📊</span>
                      <div><strong>{isRTL ? 'العوامل' : 'Top factors'}:</strong> {m.meta.top_factors?.join(', ')}</div>
                    </div>
                    {m.meta.labs && (
                      <div style={{display:'flex', gap:'0.75rem', alignItems:'center'}}>
                        <span style={{fontSize:'1.2rem'}}>🧪</span>
                        <div><strong>HbA1c:</strong> {m.meta.labs.value}{m.meta.labs.unit} ({m.meta.labs.date})</div>
                      </div>
                    )}
                    {m.meta.recommendations && (
                      <div className="card" style={{marginTop:'0.5rem'}}>
                        <div style={{fontWeight:600}}>{isRTL? 'توصيات' : 'Recommendations'} ({m.meta.recommendations.category})</div>
                        <div><strong>{isRTL? 'الغذاء' : 'Diet'}:</strong> {m.meta.recommendations.diet.join(' • ')}</div>
                        <div><strong>{isRTL? 'التمارين' : 'Workout'}:</strong> {m.meta.recommendations.workout.join(' • ')}</div>
                        <div><strong>{isRTL? 'النوم' : 'Sleep'}:</strong> {m.meta.recommendations.sleep.join(' • ')}</div>
                        <div style={{opacity:0.8, marginTop:'0.25rem'}}>{m.meta.recommendations.followup}</div>
                      </div>
                    )}
                    <div style={{marginTop:'0.5rem', display:'flex', gap:'0.5rem', flexWrap:'wrap'}}>
                      <button onClick={()=>setMessages(m=>[...m,{role:'assistant', text: isRTL? 'نصائح غذائية للمصابين بالسكري/ما قبل السكري/الصحي. اختر فئة لعرض التفاصيل.' : 'Diet suggestions for diabetic/prediabetic/healthy. Pick a category to view details.'}])}>{isRTL? 'نصائح غذائية' : 'Diet Tips'}</button>
                      <button onClick={()=>setMessages(m=>[...m,{role:'assistant', text: isRTL? 'خطة تمرين أسبوعية مبسطة وفق الحالة.' : 'Simple weekly workout plan by condition.'}])}>{isRTL? 'تمارين' : 'Workout'}</button>
                      <button onClick={()=>setMessages(m=>[...m,{role:'assistant', text: isRTL? 'إرشادات النوم: جدول ثابت، مدة كافية، عادات صحية.' : 'Sleep guidance: steady schedule, adequate duration, good hygiene.'}])}>{isRTL? 'النوم' : 'Sleep'}</button>
                      {m.meta.explain && (
                        <a href={m.meta.explain.shap_summary} target="_blank" rel="noreferrer" className="btn-like">{isRTL? 'شرح SHAP' : 'SHAP Summary'}</a>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
          <div className="composer">
            <input value={message} onChange={(e) => setMessage(e.target.value)} placeholder={isRTL ? 'اكتب رسالتك...' : 'Type your message...'} onKeyDown={(e) => { if (e.key === 'Enter') send() }} />
            <button onClick={send}>{isRTL ? 'إرسال' : 'Send'}</button>
            <label style={{ display:'inline-flex', alignItems:'center', gap:'0.4rem' }}>
              <input type="file" accept="audio/*" style={{ display:'none' }} onChange={async (e)=>{
                const f = e.target.files?.[0]; if(!f) return;
                setTranscribing(true);
                const fd = new FormData(); fd.append('audio', f);
                try{
                  const r = await fetch(`${CHAT_API}/voice/transcribe`, { method:'POST', body: fd });
                  const data = await r.json();
                  if (data?.text) setMessage(prev => (prev ? prev + ' ' + data.text : data.text));
                }catch{}
                setTranscribing(false);
              }} />
              <span className="btn-like">{transcribing ? (isRTL ? 'جارٍ النسخ...' : 'Transcribing...') : (isRTL ? 'نسخ صوت' : 'Transcribe Audio')}</span>
            </label>
          </div>
          {history.length > 1 && (
            <div style={{marginTop:'0.75rem', background:'rgba(2,6,23,0.5)', border:'1px solid rgba(148,163,184,0.2)', borderRadius:12, padding:'0.5rem'}}>
              <div style={{fontSize:'0.9rem', marginBottom:'0.25rem'}}>{isRTL ? 'تاريخ المخاطر' : 'Risk History'}</div>
              <Line
                data={{
                  labels: history.map(h=>h.t),
                  datasets: [{ label: isRTL ? 'الخطر' : 'Risk', data: history.map(h=> (h.v*100).toFixed(1)), borderColor:'#60a5fa', backgroundColor:'rgba(96,165,250,0.2)' }]
                }}
                options={{ responsive:true, plugins:{ legend:{ display:false }}}}
                height={80}
              />
            </div>
          )}
          
          {lastAssessment && (
            <button 
              className="generateReportBtn" 
              onClick={generateReport} 
              disabled={reportLoading}
            >
              {reportLoading ? (
                <span style={{display:'flex', alignItems:'center', gap:'0.5rem'}}>
                  <div className="loading"></div>
                  {isRTL ? 'إنشاء التقرير...' : 'Generating Report...'}
                </span>
              ) : (
                <span style={{display:'flex', alignItems:'center', gap:'0.5rem'}}>
                  📄 {isRTL ? 'إنشاء تقرير مفصل' : 'Generate Detailed Report'}
                </span>
              )}
            </button>
          )}
        </main>
      </div>
    </div>
  )
}

export default App
