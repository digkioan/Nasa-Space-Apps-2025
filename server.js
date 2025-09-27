// server.js  — Gemini proxy (safe: keeps API key on server)
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const API_KEY = process.env.GOOGLE_API_KEY;  // βάλε το API key σου στο .env
const PORT    = process.env.PORT || 8787;

if (!API_KEY) {
  console.error('❌ GOOGLE_API_KEY is missing. Create a .env with GOOGLE_API_KEY=YOUR_KEY');
  process.exit(1);
}

/**
 * Health/ready check (GET) — χρήσιμο για δοκιμή
 */
app.get('/api/health', (_req, res) => {
  res.json({ ok: true, model: 'gemini-1.5-flash', endpoint: '/api/gemini' });
});

/**
 * Σύντομο endpoint για ping χωρίς χρέωση
 */
app.post('/api/gemini', async (req, res) => {
  try {
    const { prompt } = req.body || {};
    if (!prompt) return res.status(400).json({ error: 'Missing prompt' });

    // Γρήγορο ping ώστε να μη χρεώνεται κλήση στο μοντέλο
    if (prompt.trim().toLowerCase() === 'ping') {
      return res.json({ text: 'pong' });
    }

    // Gemini generateContent REST call
    const url =
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${API_KEY}`;

    const body = {
      contents: [{ role: 'user', parts: [{ text: prompt }]}]
    };

    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type':'application/json' },
      body: JSON.stringify(body)
    });

    if (!r.ok) {
      const errText = await r.text();
      return res.status(500).json({ error: 'Gemini error', details: errText });
    }

    const data = await r.json();
    const text = data?.candidates?.[0]?.content?.parts
      ?.map(p => p.text)
      ?.join('') || '(no response)';

    res.json({ text });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: 'Server error' });
  }
});

app.listen(PORT, () => {
  console.log(`✅ Gemini proxy listening on http://localhost:${PORT}`);
});
