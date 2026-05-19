import { useEffect, useState } from "react";
import SearchBox from "./components/SearchBox.jsx";
import ResultGrid from "./components/ResultGrid.jsx";
import { fetchStats, searchAPI } from "./api.js";

export default function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    fetchStats().then(setStats).catch(() => {});
  }, []);

  async function onSearch(query) {
    setLoading(true);
    setError(null);
    try {
      const data = await searchAPI(query);
      setResults(data);
    } catch (e) {
      setError(e.message);
      setResults(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <header>
        <h1>BetterDay v2 — Multimedia Search</h1>
        <span className="sub">SigLIP-2 · OCR · Caption · Detection · ASR · FAISS + 2× BM25</span>
        {stats && (
          <span className="stats">
            {stats.items_video}🎥 {stats.items_audio}🎵 {stats.items_image}🖼 · {stats.frames} frames · {stats.scenes ?? 0} scenes · {stats.asr_segments} asr
          </span>
        )}
      </header>

      <SearchBox onSearch={onSearch} loading={loading} />

      {error && <div className="error">⚠ {error}</div>}

      {results && (
        <>
          <div className="meta">
            <details>
              <summary>
                {results.expanded_queries.length} biến thể · {results.num_dense} dense ·{" "}
                {results.num_bm25_visual} bm25_v · {results.num_bm25_asr} bm25_a · {results.results.length} kết quả
                {results.timing_ms && (
                  <> · {Object.values(results.timing_ms).reduce((a, b) => a + b, 0).toFixed(0)}ms</>
                )}
              </summary>
              <div className="expanded">
                <strong className="label">Dense (SigLIP):</strong>
                {results.expanded_queries.map((q, i) => (
                  <span key={`d${i}`} className="chip">{q}</span>
                ))}
              </div>
              {results.bm25_queries && results.bm25_queries.length > 0 && (
                <div className="expanded">
                  <strong className="label">BM25 (OR):</strong>
                  {results.bm25_queries.map((q, i) => (
                    <span key={`b${i}`} className="chip bm25">{q}</span>
                  ))}
                </div>
              )}
              {results.timing_ms && (
                <pre className="timing">{JSON.stringify(results.timing_ms, null, 2)}</pre>
              )}
            </details>
          </div>
          <ResultGrid results={results.results} />
        </>
      )}
    </div>
  );
}
