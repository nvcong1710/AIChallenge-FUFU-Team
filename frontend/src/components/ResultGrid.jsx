import { thumbnailURL } from "../api.js";

function formatTs(s) {
  if (s == null) return "—";
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1);
  return `${m}:${sec.padStart(4, "0")}`;
}

function basename(p) {
  return String(p || "").split(/[\\/]/).pop();
}

const TYPE_ICON = { video: "🎥", audio: "🎵", image: "🖼" };

function ScoreBreakdown({ b }) {
  if (!b) return null;
  return (
    <span className="breakdown" title="dense / bm25_visual / bm25_asr">
      D{(b.dense ?? 0).toFixed(2)} V{(b.bm25_visual ?? 0).toFixed(2)} A{(b.bm25_asr ?? 0).toFixed(2)}
    </span>
  );
}

function VideoCard({ r, rank }) {
  return (
    <div className="card">
      {r.best_frame?.thumbnail && (
        <img src={thumbnailURL(r.best_frame.thumbnail)} alt="" loading="lazy" />
      )}
      <div className="info">
        <div className="row">
          <span className="rank">{TYPE_ICON[r.media_type]} #{rank}</span>
          <span className="score">{r.score.toFixed(3)}</span>
        </div>
        <div className="ts">
          shot {formatTs(r.segment_start)} – {formatTs(r.segment_end)}
          {r.best_frame?.timestamp != null && (
            <span className="frame-ts"> · @ {formatTs(r.best_frame.timestamp)}</span>
          )}
        </div>
        {r.scene_id != null && (
          <div className="scene">
            🎬 scene {r.scene_id} · {formatTs(r.scene_start)}–{formatTs(r.scene_end)} · {r.scene_n_shots} shots
          </div>
        )}
        {r.best_frame?.caption && <div className="caption">{r.best_frame.caption}</div>}
        {r.best_asr?.text && <div className="asr">🗣 {r.best_asr.text}</div>}
        <div className="vid" title={r.item_path}>
          {basename(r.item_path)}
        </div>
        <ScoreBreakdown b={r.score_breakdown} />
      </div>
    </div>
  );
}

function ImageCard({ r, rank }) {
  return (
    <div className="card">
      {r.best_frame?.thumbnail && (
        <img src={thumbnailURL(r.best_frame.thumbnail)} alt="" loading="lazy" />
      )}
      <div className="info">
        <div className="row">
          <span className="rank">{TYPE_ICON[r.media_type]} #{rank}</span>
          <span className="score">{r.score.toFixed(3)}</span>
        </div>
        {r.best_frame?.caption && <div className="caption">{r.best_frame.caption}</div>}
        <div className="vid" title={r.item_path}>{basename(r.item_path)}</div>
        <ScoreBreakdown b={r.score_breakdown} />
      </div>
    </div>
  );
}

function AudioCard({ r, rank }) {
  return (
    <div className="card audio">
      <div className="audio-icon">🎵</div>
      <div className="info">
        <div className="row">
          <span className="rank">{TYPE_ICON[r.media_type]} #{rank}</span>
          <span className="score">{r.score.toFixed(3)}</span>
        </div>
        {r.best_asr && (
          <>
            <div className="ts">
              {formatTs(r.best_asr.start)} – {formatTs(r.best_asr.end)}
            </div>
            <div className="asr">🗣 {r.best_asr.text}</div>
          </>
        )}
        <div className="vid" title={r.item_path}>{basename(r.item_path)}</div>
        <ScoreBreakdown b={r.score_breakdown} />
      </div>
    </div>
  );
}

export default function ResultGrid({ results }) {
  if (!results || results.length === 0) {
    return <div className="empty">Không có kết quả nào khớp.</div>;
  }
  return (
    <div className="grid">
      {results.map((r, i) => {
        const props = { r, rank: i + 1 };
        if (r.media_type === "audio") return <AudioCard key={`${r.item_id}-${r.segment_id ?? "x"}`} {...props} />;
        if (r.media_type === "image") return <ImageCard key={`${r.item_id}-${r.segment_id ?? "x"}`} {...props} />;
        return <VideoCard key={`${r.item_id}-${r.segment_id ?? "x"}`} {...props} />;
      })}
    </div>
  );
}
