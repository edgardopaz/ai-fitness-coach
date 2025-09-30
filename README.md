# AI Form Coach

AI Form Coach pairs MediaPipe pose tracking with Gemini-generated coaching cues to deliver real-time form feedback for common bodyweight movements. This project targets quick iteration during OwlHacks and is built with React, TypeScript, and Vite.

## Prerequisites
- Node.js 20.x or newer (matching the Vite 7 toolchain)
- npm 10.x or newer
- A webcam and browser that support `navigator.mediaDevices.getUserMedia`

## Environment Variables
Create a `.env` file (or update the existing one) in the project root and define the Gemini key that powers the rotating coaching copy.

```
VITE_GEMINI_API_KEY=your_api_key_here
```

Leave the value blank if you want to skip the live callout text; the rest of the app will continue to function.

## Install Dependencies
```
npm install
```

## Run Locally
```
npm run dev
```
Open the URL printed in the console (defaults to http://localhost:5173) and grant camera permission so MediaPipe can stream frames into the pose coach.

## Build For Production
```
npm run build
```
This command performs a TypeScript project build (`tsc -b`) followed by `vite build`. The compiled static site is written to `dist/`. Run the optional preview server to sanity check the production bundle:

```
npm run preview
```

## Update The Live Site
1. Run `npm run build` to generate an up-to-date `dist/` folder.
2. Smoke test the bundle locally with `npm run preview` (hit http://localhost:4173).
3. Deploy the contents of `dist/` to your hosting provider. For static hosts this usually means copying the files to the server root (S3 bucket, Azure Static Web Apps, Netlify drop, etc).
4. If you publish via GitHub Pages, one straightforward option is to push the `dist/` folder to a `gh-pages` branch. For example:

```
git add dist -f
git commit -m "Deploy latest dist"
git subtree push --prefix dist origin gh-pages
```

Afterwards, reset the temporary commit if you do not want to keep `dist/` tracked.
5. Clear any CDN caches so the new assets are served immediately.

## Helpful Scripts
- `npm run dev` - starts the Vite dev server with hot module replacement.
- `npm run build` - type-checks and produces the production build inside `dist/`.
- `npm run preview` - serves the last production build locally.
- `npm run lint` - runs ESLint across the codebase.

# Rebuild Build Site Script just add folder 
sudo rsync -av --delete dist/ /var/www/{folderName}
sudo chown -R www-data:www-data /var/www/{folderName}
sudo find /var/www/{folderName} -type d -exec chmod 755 {} \;
sudo find /var/www/{folderName} -type f -exec chmod 644 {} \;

## Troubleshooting Notes
- If MediaPipe fails to initialize, reload the page after confirming camera permissions were granted.
- Generative copy falls back to static lines whenever `VITE_GEMINI_API_KEY` is missing or the Gemini API returns an error.
- For best results keep the athlete framed head-to-toe with good lighting so pose landmarks stay stable.
