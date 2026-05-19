const BASE = import.meta.env.VITE_API_BASE || "http://localhost:8080";

export async function searchAPI(query, topK = 20) {
  const res = await fetch(`${BASE}/api/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  return res.json();
}

export async function fetchStats() {
  const res = await fetch(`${BASE}/api/stats`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export function thumbnailURL(thumbnailPath) {
  // thumbnailPath là absolute server path; lấy phần sau "/thumbnails/"
  const norm = String(thumbnailPath).replace(/\\/g, "/");
  const idx = norm.lastIndexOf("/thumbnails/");
  if (idx === -1) {
    const parts = norm.split("/");
    return `${BASE}/thumbnails/${parts.slice(-2).join("/")}`;
  }
  return `${BASE}${norm.slice(idx)}`;
}

export const API_BASE = BASE;
