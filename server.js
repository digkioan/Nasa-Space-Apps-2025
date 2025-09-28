// server.js  (CommonJS)
const express = require("express");
const cors = require("cors");
require("dotenv").config();

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

const API_KEY = process.env.GROQ_API_KEY;
const PORT = process.env.PORT || 8787;

if (!API_KEY) {
  console.error("❌ GROQ_API_KEY missing in .env");
  process.exit(1);
}

// English-only guard
function isAsciiOnly(s){ return /^[\x00-\x7F]*$/.test(s); }

// Proxy endpoint
app.post("/api/groq", async (req, res) => {
  try {
    const { prompt } = req.body || {};
    if (!prompt) return res.status(400).json({ error: "Missing prompt" });
    if (!isAsciiOnly(prompt)) return res.status(400).json({ error: "English letters only" });

    const r = await fetch("https://api.groq.com/openai/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + API_KEY
      },
      body: JSON.stringify({
        model: "llama-3.1-8b-instant", // ή "llama-3.3-70b-versatile"
        messages: [
          { role: "system", content: "Be brief. Answer in 2-4 lines. English only." },
          { role: "user", content: prompt }
        ],
        temperature: 0.2,
        max_tokens: 180
      })
    });

    const j = await r.json();
    if (!r.ok) return res.status(500).json({ error: j?.error?.message || "Groq error" });
    res.json({ text: j?.choices?.[0]?.message?.content ?? "(no response)" });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Server error" });
  }
});

// Σέρβιρε τα τοπικά αρχεία σου (map.html, server.html κτλ) από τον φάκελο του project
const path = require("path");
app.use(express.static(path.resolve("."), { dotfiles: "ignore", extensions: ["html"] }));

app.listen(PORT, () => console.log(`✅ Running on http://localhost:${PORT}  (open /map.html)`));
