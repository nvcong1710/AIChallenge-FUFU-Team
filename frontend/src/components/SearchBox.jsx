import { useState } from "react";

export default function SearchBox({ onSearch, loading }) {
  const [q, setQ] = useState("");

  function submit(e) {
    e.preventDefault();
    const v = q.trim();
    if (v) onSearch(v);
  }

  return (
    <form className="searchbox" onSubmit={submit}>
      <input
        type="text"
        placeholder="Mô tả cảnh muốn tìm... (VD: người phụ nữ áo đỏ cầm ô)"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        disabled={loading}
        autoFocus
      />
      <button type="submit" disabled={loading || !q.trim()}>
        {loading ? "Đang tìm..." : "Tìm"}
      </button>
    </form>
  );
}
