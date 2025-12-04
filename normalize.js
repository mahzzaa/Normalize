// normalize.js
const fs = require("fs");
const path = require("path");

// ---------- Helper: normalize an array to [0, 1] ----------
function normalizeArray(arr) {
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const range = max - min || 1; // avoid division by zero if all values are equal
  return arr.map(v => (v - min) / range);
}

// ---------- Helper: normalize CSV content ----------
function normalizeCsv(csvString) {
  // Split into non-empty lines
  const lines = csvString
    .split("\n")
    .map(l => l.trim())
    .filter(l => l.length > 0);

  if (lines.length === 0) {
    throw new Error("CSV file is empty");
  }

  const headers = lines[0].split(",").map(h => h.trim());

  const timestampIndex = headers.indexOf("Timestamp");
  const xIndex = headers.indexOf("X");
  const yIndex = headers.indexOf("Y");
  const zIndex = headers.indexOf("Z");

  if (xIndex === -1 || yIndex === -1 || zIndex === -1) {
    throw new Error("CSV must have X, Y, Z columns");
  }

  const rows = lines.slice(1).map(line => line.split(","));
  console.log(`Processing ${rows.length} data rows...`);

  // Extract numeric columns
  const xs = rows.map(r => parseFloat(r[xIndex]));
  const ys = rows.map(r => parseFloat(r[yIndex]));
  const zs = rows.map(r => parseFloat(r[zIndex]));

  // Normalize X, Y, Z
  const xsNorm = normalizeArray(xs);
  const ysNorm = normalizeArray(ys);
  const zsNorm = normalizeArray(zs);

  // Add new header columns
  const newHeaders = [...headers, "X_norm", "Y_norm", "Z_norm"];

  // Build new CSV rows
  const newRows = rows.map((row, i) => {
    const xNorm = xsNorm[i];
    const yNorm = ysNorm[i];
    const zNorm = zsNorm[i];

    return [
      ...row,
      xNorm.toString(),
      yNorm.toString(),
      zNorm.toString(),
    ].join(",");
  });

  // Join headers + rows back to CSV string
  const outputCsv = [newHeaders.join(","), ...newRows].join("\n");

  return outputCsv;
}

// ---------- Main: read file, normalize, write new file ----------
const inputFile = path.join(__dirname, "sensor_data_14th_may_1600.csv");
const outputFile = path.join(__dirname, "sensor_data_14th_may_1600_normalized.csv");

fs.readFile(inputFile, "utf8", (err, data) => {
  if (err) {
    console.error("Error reading input CSV:", err.message);
    process.exit(1);
  }

  try {
    const normalizedCsv = normalizeCsv(data);
    fs.writeFile(outputFile, normalizedCsv, "utf8", err2 => {
      if (err2) {
        console.error("Error writing output CSV:", err2.message);
        process.exit(1);
      }

      console.log("✅ Normalized data saved to:", outputFile);
    });
  } catch (e) {
    console.error("Error normalizing CSV:", e.message);
    process.exit(1);
  }
});

// Optional: export helpers if you want to import them elsewhere
module.exports = {
  normalizeArray,
  normalizeCsv,
};
