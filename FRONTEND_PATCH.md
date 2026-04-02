# Front-end patch for the current HTML UI

Your backend accepts all of these fields:

- `target_lufs`
- `warmth`
- `brightness`
- `punch`
- `low_eq`
- `mid_eq`
- `high_eq`
- `compression`

Your current HTML only sends the first four. Replace your `getMasteringSettings()` function with:

```javascript
function getMasteringSettings() {
  return {
    target_lufs: document.getElementById("targetLufs").value,
    warmth: document.getElementById("warmth").value,
    brightness: document.getElementById("brightness").value,
    punch: document.getElementById("punch").value,
    low_eq: document.getElementById("lowEq").value,
    mid_eq: document.getElementById("midEq").value,
    high_eq: document.getElementById("highEq").value,
    compression: document.getElementById("compression").value,
  };
}
```

If your backend is hosted on a different domain than the page, replace:

```javascript
const apiBase = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : window.location.origin;
```

with something like:

```javascript
const apiBase = "https://your-backend-domain.com";
```

If you keep the backend on another origin, make sure `BSAI_ALLOWED_ORIGINS` includes your Wix site URL.
