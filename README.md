# NASA Space Apps 2025 — Air Quality Globe + Country Stats

Interactive web app that visualizes air-quality elements on a 3D globe and computes a simple “overall score” per country using **live data** (OpenAQ + World Bank). It also includes a lightweight AI chat (via Groq) and multiple statistics dashboards.

## ✨ Highlights

* **3D Globe** (NASA Web WorldWind) with overlays for **O₃, NO₂, SO₂, CO, PM₁₀, PM₂.₅**

  * Time slider, auto-play, legend with robust min/max, joystick panning
* **Country “Overall Score”** (client.html)

  * Pulls **PM2.5** from OpenAQ (converted to AQI) and **Life expectancy** from World Bank
  * Donut meter + KPI tiles, plus three histograms:

    * Avg house price (demo dataset)
    * Avg life-insurance premium (demo dataset)
    * Life expectancy (live), with tight Y-axis to emphasize small differences
* **AI Chat** (map.html & client.html)

  * Proxy endpoint `/api/groq` (no keys in client)
  * Health check `/api/health`
* Simple **start menu** (start.html) with links to **MAP**, **STATISTICS**, **ABOUT**

---

## 🗂 Project structure

```
Nasa-Space-Apps-2025/
├── start.html        # Landing page (animated stars, moon, buttons)
├── map.html          # 3D Globe + overlays + AI chat
├── client.html       # Country score + histograms + AI chat
├── server.js         # Express server + /api/groq proxy + /api/health
├── .env              # GROQ_API_KEY=... (NOT committed)
├── o3_*.csv, no2_*.csv, so2_*.csv, co_*.csv, pm10_*.csv, pm25_*.csv  # overlay data
├── gif_background.gif, stars.gif, moon.gif                           # visuals
├── package.json
└── README.md
```

---

## 🛠 Tech stack

* **Frontend:** Vanilla HTML/CSS/JS, Chart.js
* **3D Globe:** **NASA Web WorldWind** (`worldwind.min.js`)
* **APIs (live):**

  * **OpenAQ v3** for PM2.5 → AQI
  * **World Bank** for life expectancy (SP.DYN.LE00.IN)
* **Backend:** Node.js + Express
* **LLM:** Groq (OpenAI-compatible Chat Completions) → proxied via `/api/groq`

---

## ▶️ Run locally

**Requirements:** Node.js 18+.

1. Install deps

```bash
npm install
```

2. Create `.env` (never commit this file):

```bash
GROQ_API_KEY=YOUR_GROQ_KEY_HERE
PORT=8787
```

3. Start server

```bash
npm start
# ✅ Running on http://localhost:8787  (open /map.html)
```

4. Open in the browser:

* `http://localhost:8787/start.html` (menu)
* `http://localhost:8787/map.html` (3D globe + overlays + chat)
* `http://localhost:8787/client.html` (country score + histograms + chat)

---

## 🔌 API endpoints (local)

* `GET /api/health` → `{ ok: true, hasKey: boolean }`
* `POST /api/groq`

  ```json
  {
    "model": "llama-3.1-8b-instant",
    "prompt": "Hello!"
  }
  ```

  Response:

  ```json
  { "text": "Hello ..." }
  ```

> The browser never sees your API key. The frontend calls `/api/groq`, and the Node server injects `GROQ_API_KEY` from `.env`.

---

## 🔐 Secrets & GitHub push protection

* **Do not commit** `.env`. It’s git-ignored.
* If GitHub blocks a push (secret scanning), rewrite the commit that contains the secret:

  ```bash
  git reset --soft <safe_commit>
  git restore --staged .env
  git commit -m "Remove secrets"
  git push --force-with-lease
  ```

---

## 🧮 Scoring (client.html)

* Fetches **PM2.5** (OpenAQ) → converts to US AQI via piecewise linear mapping.
* Fetches **Life expectancy** (World Bank).
* Computes **Overall score**:

  ```
  score = (1 − w * penalty(AQI)) * QoL%
  ```

  with `w = 0.4` and `penalty(AQI) ∈ [0..1]`.
  The donut color and face emoji change with the score.

---

## 📊 Charts

* **Life expectancy** histogram has **tight y-axis** (min/max with margin) to reveal small differences (78–83 years).
* Current country bar is **highlighted in green**.
* Price & insurance datasets are curated demo values (static maps in code).

---

## 🧭 Controls (map.html)

* **Year** selector, **Date** slider, **Auto-play**.
* **Element buttons:** O₃, NO₂, SO₂, CO, PM₁₀, PM₂.₅.
* **Legend** with dynamic robust min/max (1st–99th percentiles).
* **Joystick** for panning the globe.
* **Double-click** on the globe to ask AI for a short location description.

---

## 🧪 Troubleshooting

* **HTTP 405 / AI chat**
  Ensure you’re opening pages **through the Node server** ([http://localhost:8787](http://localhost:8787)), not from the file system.
  `/api/groq` only exists on the server.
* **“GROQ_API_KEY is missing”**
  Create `.env` with your key and restart `npm start`.
* **Port in use**
  Change `PORT` in `.env` or free the port.
* **CORS**
  The Express app enables CORS for local use.

---

## 🚀 Deploy

* Any Node host (Render, Railway, Fly, etc.).
* Set environment variable: `GROQ_API_KEY`.
* Serve static files from the project root (the server already does `express.static(__dirname)`).

---

## 📌 Roadmap

* Add wildfire smoke layers
* Bias-correction between satellite & ground sensors
* User thresholds & geo-alerts
* Country comparison & bookmarks
* Export images / CSV

---

## 🙏 Acknowledgments

* **NASA Web WorldWind**
* **OpenAQ** (community air-quality data)
* **World Bank** (indicators)
* **Groq** (LLM API)

---

## 📝 License

ISC (see `LICENSE` if present).
Data sources may have their own terms—verify before reuse.

---

## 👤 Authors

Chillers — NASA Space Apps 2025 hackathon project.
Questions/feedback: open an issue on the repo.

