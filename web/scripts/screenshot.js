import puppeteer from 'puppeteer';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const url = process.env.APP_URL || 'http://127.0.0.1:5173';
const outDir = path.resolve(__dirname, '../../slides');

const shots = [
  { name: 'home_en.png', lang: 'en' },
  { name: 'home_ar.png', lang: 'ar' },
];

const run = async () => {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  for (const s of shots) {
    await page.goto(url);
    await page.waitForSelector('header');
    // set language via select
    await page.select('header select', s.lang);
    await page.screenshot({ path: path.join(outDir, s.name), fullPage: true });
  }
  await browser.close();
};

run().catch((e) => { console.error(e); process.exit(1); });


