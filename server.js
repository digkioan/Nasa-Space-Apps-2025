// server.js  (Node 18+, ESM)
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const app  = express();
const PORT = process.env.PORT || 8787;

app.use(cors());
app.use(express.json());

// Σερβίρισμα static (map.html κ.λπ.)
app.use(express.static(__dirname, { extensions: ['html'] }));

// Healthcheck
app.get('/api/health', (_req, res) => {
  res.json({ ok: true, hasKey: Boolean(process.env.GROQ_API_KEY) });
});

// ---- Proxy προς Groq ----
app.post('/api/groq', async (req, res, next) => {
  try {
    const key     = process.env.GROQ_API_KEY;
    const model   = (req.body?.model || 'llama-3.1-8b-instant').trim();
    const prompt  = (req.body?.prompt || '').trim();
    if (!key)    return res.status(500).json({ error: 'GROQ_API_KEY is missing in .env' });
    if (!prompt) return res.status(400).json({ error: 'Empty prompt' });

    const headers = { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key };

    // 1) Προσπάθεια με chat/completions (OpenAI-compatible)
    let r = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST', headers,
      body: JSON.stringify({
        model,
        messages: [
          { role: 'system', content: 'You are a helpful assistant. Answer in concise English (1–2 sentences).' },
          { role: 'user',   content: prompt }
        ],
        temperature: 0.2,
        max_tokens: 140
      })
    });

    // 405/404 -> fallback στο Responses API
    if (r.status === 404 || r.status === 405) {
      const r2 = await fetch('https://api.groq.com/openai/v1/responses', {
        method: 'POST', headers,
        body: JSON.stringify({
          model,
          input: [
            { role: 'system', content: [{ type: 'text', text: 'You are a helpful assistant. Answer in concise English (1–2 sentences).' }] },
            { role: 'user',   content: [{ type: 'text', text: prompt }] }
          ],
          temperature: 0.2,
          max_output_tokens: 140
        })
      });
      const d2 = await r2.json().catch(()=>null);
      if (!r2.ok) return res.status(r2.status).json({ error: d2?.error?.message || d2?.error || 'Groq responses error' });
      const out = d2?.output?.[0]?.content?.[0]?.text ?? d2?.output_text ?? '';
      return res.json({ text: out || '(empty response)' });
    }

    // Κανονικό path (chat/completions)
    const d = await r.json().catch(()=>null);
    if (!r.ok) return res.status(r.status).json({ error: d?.error?.message || d?.error || 'Groq chat error' });
    const out = d?.choices?.[0]?.message?.content ?? '';
    return res.json({ text: out || '(empty response)' });

  } catch (err) {
    next(err);
  }
});

// JSON error handler
app.use((err, _req, res, _next) => {
  console.error('SERVER ERROR:', err);
  res.status(500).json({ error: err?.message || 'Internal server error' });
});

app.listen(PORT, () => {
  console.log(`✅ Running on http://localhost:${PORT}  (open /map.html)`);
});
