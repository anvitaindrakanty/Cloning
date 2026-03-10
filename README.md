<div align="center">

<img src="https://33uee5uclf.ufs.sh/f/fa600f97-6cf2-4759-8641-3093d13591bf-6rrk6b.png" alt="Resonance" width="720" />

<br />
<br />

<h1>Resonance</h1>

<p>The open-source ElevenLabs alternative.</p>

<p>AI-powered text-to-speech and voice cloning built with Next.js 16, React 19, and Chatterbox TTS.</p>

<br />

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/SioRb1?referralCode=ANTONIO&utm_medium=integration&utm_source=template&utm_campaign=generic)

<br />

<p>
  <a href="https://cwa.run/clerk"><img src="https://img.shields.io/badge/Clerk-6C47FF?style=for-the-badge&logo=clerk&logoColor=white" alt="Clerk" /></a>&nbsp;
  <a href="https://cwa.run/polar"><img src="https://img.shields.io/badge/Polar-000000?style=for-the-badge&logo=polar&logoColor=white" alt="Polar" /></a>&nbsp;
  <a href="https://cwa.run/railway"><img src="https://img.shields.io/badge/Railway-0B0D0E?style=for-the-badge&logo=railway&logoColor=white" alt="Railway" /></a>&nbsp;
  <a href="https://cwa.run/sentry"><img src="https://img.shields.io/badge/Sentry-362D59?style=for-the-badge&logo=sentry&logoColor=white" alt="Sentry" /></a>&nbsp;
  <a href="https://cwa.run/coderabbit"><img src="https://img.shields.io/badge/CodeRabbit-FF6C37?style=for-the-badge&logo=rabbitmq&logoColor=white" alt="CodeRabbit" /></a>&nbsp;
  <a href="https://cwa.run/prisma"><img src="https://img.shields.io/badge/Prisma-2D3748?style=for-the-badge&logo=prisma&logoColor=white" alt="Prisma" /></a>
</p>

</div>

<br />

## Tutorial

[![Watch on YouTube](https://img.shields.io/badge/Watch_the_Full_Course-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://cwa.run/resonance-gh-yt)

Learn how to build this entire project from scratch in a **free 12-hour video course** on YouTube. The tutorial covers every feature  - authentication, text-to-speech, voice cloning, billing, deployment, and more.

Each chapter has a matching branch so you can check out the code at any point in the tutorial:

| Branch | Chapter |
|--------|---------|
| `main` | Final project (all chapters combined) |
| `02-dashboard` | Dashboard layout and navigation |
| `03-text-to-speech-ui` | Text-to-speech UI |
| `04-backend-infrastructure` | Backend infrastructure (tRPC, R2, Prisma) |
| `05-voice-selection` | Voice selection and library |
| `06-tts-generation-audio-player` | TTS generation and audio player |
| `07-tts-history-polish` | TTS history and polish |
| `bonus-sentry-error-monitoring` | Bonus: Sentry error monitoring |
| `08-voice-management` | Voice management and cloning |
| `09-billing` | Billing and usage metering |

```bash
git checkout 04-backend-infrastructure  # example: jump to Chapter 4
```

## Features

- **Text-to-Speech**  - Generate speech from text with adjustable creativity, variety, expression, and flow parameters
- **Zero-Shot Voice Cloning**  - Upload or record a voice sample (10s minimum) and clone it instantly  - no fine-tuning required
- **20 Built-in Voices**  - Pre-seeded system voices across 12 categories and 5 locales
- **Waveform Audio Player**  - WaveSurfer.js visualization with seek, play/pause, and download
- **Multi-Tenant**  - Team-based access via Clerk Organizations with full data isolation
- **Usage-Based Billing**  - Pay-as-you-go character metering with configurable pricing via Polar products and meters
- **Generation History**  - Browse and replay past generations with preserved voice metadata
- **Fully Responsive**  - Mobile-first with bottom drawers, compact controls, and adaptive layouts

## Getting Started

### Prerequisites

- Node.js **20.9** or later
- [Prisma Postgres](https://cwa.run/prisma) database
- [Clerk](https://cwa.run/clerk) account (with Organizations enabled)
- [Cloudflare R2](https://cwa.run/cloudflare-r2) bucket
- [Modal](https://cwa.run/modal) account (for GPU-hosted TTS)
- [Polar](https://cwa.run/polar) account (for billing)

### 1. Clone and install

```bash
git clone https://github.com/code-with-antonio/resonance.git
cd resonance
npm install
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in the blank values in `.env`. Sensible defaults (Clerk routes, Polar meter names, `APP_URL`, etc.) are pre-filled.

### 3. Set up Polar billing

In your [Polar](https://cwa.run/polar) dashboard, create two **meters** under **Meters**:

1. **Voice Creation** meter
   - Filter: Name equals `voice_creation`
   - Aggregation: **Count**

2. **Text-to-Speech Characters** meter
   - Filter: Name equals `tts_generation`
   - Aggregation: **Sum** over `characters`

Then create a new **product** with **Recurring subscription** pricing. Under **Price Type**, add two metered prices:

1. Click **Add metered price** and select the **Text-to-Speech Characters** meter
   - Set the **Amount per unit** (price per character, e.g. `$0.003`)
   - Optionally set a **Cap amount** (e.g. `$100`)

2. Click **Add metered price** again and select the **Voice Creation** meter
   - Set the **Amount per unit** (price per voice generation, e.g. `$0.25`)
   - Optionally set a **Cap amount** (e.g. `$100`)

With only metered prices, the subscription starts at **$0/month** and scales with usage. If you want a baseline subscription fee (e.g. $20/month), add a third price to the same product — select a **fixed price** instead of a metered price. This requires no code changes since fixed prices are handled entirely by Polar.

Ensure **Allow multiple subscriptions** is turned **off** under **Settings > Billing** (this is the Polar default).

Copy the product ID into `POLAR_PRODUCT_ID`. The meter filter names and aggregation property must match the `POLAR_METER_*` env variables.

### 4. Set up the database

```bash
npx prisma migrate deploy
```

### 5. Deploy the TTS engine

The included `chatterbox_tts.py` is adapted from [Modal's official Chatterbox TTS example](https://cwa.run/modal-tts), modified to read voice reference audio directly from your R2 bucket instead of a Modal Volume.

Before deploying, update `chatterbox_tts.py` with your R2 credentials:

```python
R2_BUCKET_NAME = "<your-r2-bucket-name-here>"
R2_ACCOUNT_ID = "<your-r2-account-id-here>"
```

Then create the required secrets in your [Modal dashboard](https://cwa.run/modal-secrets):

| Secret Name | Keys | Description |
|-------------|------|-------------|
| `cloudflare-r2` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | R2 API credentials (used for bucket mount) |
| `chatterbox-api-key` | `CHATTERBOX_API_KEY` | API key to protect the endpoint (use any strong random string) |
| `hf-token` | `HF_TOKEN` | Hugging Face token (for downloading the Chatterbox model weights) |

Deploy to Modal:

```bash
modal deploy chatterbox_tts.py
```

This deploys Chatterbox TTS to a serverless NVIDIA A10G GPU on Modal. The container mounts your R2 bucket read-only for direct access to voice reference audio. Use the resulting Modal URL as `CHATTERBOX_API_URL` in your `.env.local`.

> **Note:** The first request after a period of inactivity may take longer due to cold starts as Modal provisions the GPU container.

Once deployed, generate the type-safe Chatterbox client from the OpenAPI spec:

```bash
npm run sync-api
```

### 6. Seed voices

```bash
npx prisma db seed
```

Seeds 20 built-in voices to the database and R2. The system voice WAV files are included in the repository and originate from [Modal's voice sample pack](https://modal-cdn.com/blog/audio/chatterbox-tts-voices.zip).

### 7. Run

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Self-Hosting

Resonance is designed to be self-hosted. You'll need:

1. **A PostgreSQL database**  - [Prisma Postgres](https://cwa.run/prisma) (recommended), or any managed Postgres
2. **Cloudflare R2**  - For audio storage (S3-compatible, generous free tier)
3. **Modal**  - For serverless GPU inference (pay-per-second billing)
4. **Clerk**  - For authentication and multi-tenancy
5. **Polar**  - For metered billing (use sandbox mode with card `4242 4242 4242 4242` for testing)

Deploy the Next.js app to any Node.js host (Railway, Docker, etc.).

## Project Structure

```
src/
├── app/                        # Next.js App Router
│   ├── (dashboard)/            # Protected routes (home, TTS, voices)
│   ├── api/                    # Audio proxy routes + tRPC handler
│   ├── sign-in/                # Clerk auth pages
│   └── sign-up/
├── components/                 # Shared UI components (shadcn/ui + custom)
├── features/
│   ├── dashboard/              # Home page, quick actions
│   ├── text-to-speech/         # TTS form, audio player, settings, history
│   ├── voices/                 # Voice library, creation, recording
│   └── billing/                # Usage display, checkout
├── hooks/                      # App-wide hooks
├── lib/                        # Core: db, r2, polar, env, chatterbox client
├── trpc/                       # tRPC routers, client, server helpers
├── generated/                  # Prisma client
└── types/                      # Generated API types
```

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start dev server |
| `npm run build` | Production build |
| `npm run start` | Start production server |
| `npm run lint` | Lint with ESLint |
| `npm run sync-api` | Regenerate Chatterbox API types from OpenAPI spec |

## Acknowledgements

- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) by Resemble AI - the open-source zero-shot voice cloning model powering speech generation
- [Modal](https://cwa.run/modal-tts) - serverless GPU deployment example and [voice sample pack](https://modal-cdn.com/blog/audio/chatterbox-tts-voices.zip)
