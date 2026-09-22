# ClippyMe — Public SaaS Launch Guide (Path B: €20/Month Hybrid)

This guide walks you through taking ClippyMe public as a production SaaS application within a strict **€20/month budget**.

---

## 1. Architecture & Cost Breakdown

| Component | Provider & Tier | Monthly Cost | Free Tier Capacity |
|---|---|---|---|
| **Frontend CDN** | Cloudflare Pages | **€0.00** | Unlimited bandwidth & requests |
| **User Auth & DB** | Supabase (PostgreSQL) | **€0.00** | 50,000 monthly active users (MAU) |
| **Clip Storage & Streaming** | Cloudflare R2 | **€0.00 – €1.50** | 10 GB free storage, **€0 egress fees** |
| **API Web Server** | Hetzner Cloud CX22 / Fly.io | **€4.50 – €5.00** | 2 vCPU, 4GB RAM, 40GB NVMe SSD, 20TB traffic |
| **Video Rendering & AI Tracking** | Modal.com (Nvidia T4 GPU) | **€0.00** | **$30.00 recurring credit free each month** (~1,000–3,000 mins GPU) |
| **AI Viral Detection** | Google Gemini 2.5 Flash | **€0.00 – €3.00** | Free tier (15 RPM) or pay-as-you-go (~$0.02 per long video) |
| **Total Estimated Cost** | | **~€4.50 – €9.50 / month** | **Over 50% under the €20 budget cap!** |

---

## 2. Step-by-Step Setup

### Step 1: Set Up Supabase (Auth)
1. Go to [database.new](https://database.new) and create a free project.
2. In **Project Settings → API**:
   - Copy `Project URL` (e.g. `https://xyzcompany.supabase.co`).
   - Copy `anon` public key.
   - Copy `JWT Secret`.
3. In **Authentication → Providers → Email**:
   - Ensure "Enable Email provider" is turned on.
   - Turn off "Confirm email" if you want instant one-click signup without email verification delays.

### Step 2: Set Up Cloudflare R2 (Storage & CDN)
1. In the Cloudflare Dashboard, go to **R2 → Create bucket** and name it `clippyme-clips`.
2. Under bucket settings:
   - Enable **Public Access** (either use the free `*.r2.dev` subdomain or connect your custom domain like `cdn.yourdomain.com`).
3. In **R2 → Manage R2 API Tokens**:
   - Click **Create API token** with Object Read & Write permissions.
   - Note down: `Access Key ID`, `Secret Access Key`, and your `Account ID`.

### Step 3: Deploy Modal GPU Worker (Zero-Cost GPU Computing)
1. Install Modal CLI locally:
   ```bash
   pip install modal
   modal setup
   ```
2. Create secrets in your Modal dashboard or via CLI:
   ```bash
   modal secret create clippyme-secrets GEMINI_API_KEY="your-gemini-key"
   ```
3. Deploy the GPU worker:
   ```bash
   modal deploy deploy/modal_pipeline.py
   ```
   *The worker now auto-scales to Nvidia T4 GPUs on demand and scales to zero when idle ($0 idle cost).*

### Step 4: Deploy the FastAPI Backend (Hetzner or Fly.io)
1. Provision a Hetzner Cloud CX22 instance (€4.50/mo, Ubuntu 24.04) or a Fly.io machine.
2. Clone the repository and configure `.env`:
   ```bash
   AUTH_ENABLED=1
   SUPABASE_JWT_SECRET="your-supabase-jwt-secret"

   R2_ENABLED=1
   R2_ACCOUNT_ID="your-cloudflare-account-id"
   R2_ACCESS_KEY_ID="your-r2-access-key-id"
   R2_SECRET_ACCESS_KEY="your-r2-secret-access-key"
   R2_BUCKET_NAME="clippyme-clips"
   R2_PUBLIC_DOMAIN="https://cdn.yourdomain.com"
   R2_AUTO_CLEANUP=1

   MODAL_ENABLED=1
   BILLING_ENABLED=1
   FREE_TIER_MONTHLY_MINUTES=60
   GEMINI_API_KEY="your-gemini-key"
   ```
3. Run with Docker Compose:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
   ```

### Step 5: Deploy Frontend to Cloudflare Pages (Free)
1. In Cloudflare Dashboard, go to **Workers & Pages → Create application → Pages → Connect to Git**.
2. Build configuration:
   - Framework preset: `Vite`
   - Root directory: `dashboard`
   - Build command: `npm run build`
   - Output directory: `dist`
3. Environment variables:
   - `VITE_SUPABASE_URL`: `https://your-project.supabase.co`
   - `VITE_SUPABASE_ANON_KEY`: `your-anon-key`
4. Click **Save and Deploy**. Your frontend is live worldwide on Cloudflare's global edge network!

---

## 3. Quotas & Monetization (Stripe)

- Users can register, log in, and receive **60 free minutes/month**.
- When they want to upgrade:
  - Add `STRIPE_SECRET_KEY`, `STRIPE_PRO_PRICE_ID`, and `STRIPE_WEBHOOK_SECRET` in `.env`.
  - When users upgrade to Pro, Stripe webhooks instantly grant them unlimited video processing and priority access.
