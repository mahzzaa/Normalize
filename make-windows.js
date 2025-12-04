// make_windows.js

// Node core modules:
// fs   = read/write files
// path = build file paths in a cross-platform safe way
const fs = require("fs");
const path = require("path");

/**
 * Convert a normalized CSV into multiple sequence windows for LSTM.
 *
 * @param {string} csvString - CSV content as a string
 * @param {number} seqLen    - length of each window for LSTM (e.g. 100 samples)
 * @param {number} step      - window stride (overlap). e.g. 10 means
 *                             move the window 10 samples forward each time
 * @returns {number[][][]}   - array of windows with shape [numWindows][seqLen][3]
 */
function makeWindowsFromNormalizedCsv(csvString, seqLen = 100, step = 10) {
  // 1) Split CSV string into lines and remove empty lines
  const lines = csvString
    .split("\n")         // split by newline
    .map(l => l.trim())  // trim whitespace at start/end
    .filter(l => l.length > 0); // keep only non-empty lines

  // If CSV is empty, throw an error
  if (lines.length === 0) {
    throw new Error("CSV file is empty");
  }

  // 2) First line of CSV = header row
  const headers = lines[0].split(",").map(h => h.trim());

  // Find indexes of normalized columns
  const xNormIndex = headers.indexOf("X_norm");
  const yNormIndex = headers.indexOf("Y_norm");
  const zNormIndex = headers.indexOf("Z_norm");

  // If normalized columns are missing, throw an error
  if (xNormIndex === -1 || yNormIndex === -1 || zNormIndex === -1) {
    throw new Error("CSV must have X_norm, Y_norm, Z_norm columns");
  }

  // 3) Convert each data row (except header) into a vector [x_norm, y_norm, z_norm]
  const sequence = lines.slice(1).map(line => {
    const parts = line.split(",");
    return [
      parseFloat(parts[xNormIndex]), // X_norm value
      parseFloat(parts[yNormIndex]), // Y_norm value
      parseFloat(parts[zNormIndex]), // Z_norm value
    ];
  });

  // Total number of samples
  const total = sequence.length;

  // If we don't have enough samples for at least one full window, throw an error
  if (total < seqLen) {
    throw new Error(
      `Not enough samples (${total}) for seqLen=${seqLen}`
    );
  }

  // 4) Build sliding windows
  const windows = [];

  // Start at index 0 and move forward by "step" each time
  // Stop when there is no room for a full window
  for (let start = 0; start + seqLen <= total; start += step) {
    // Slice from start to start + seqLen (end index is exclusive)
    const window = sequence.slice(start, start + seqLen); // shape: [seqLen, 3]
    windows.push(window);
  }

  // Return array of windows
  // Final shape: [numWindows][seqLen][3]
  return windows;
}

// ---------- Main / script entry point below ----------

// Input file: normalized CSV that you created in normalize.js
const inputFile = path.join(__dirname, "sensor_data_14th_may_1600_normalized.csv");
// Output file: windows saved as JSON
const outputFile = path.join(__dirname, "sensor_data_14th_may_1600_windows.json");

// You can change seqLen and step later if you want
const SEQ_LEN = 100;
const STEP = 10;

// Read the normalized CSV file from disk
fs.readFile(inputFile, "utf8", (err, data) => {
  if (err) {
    console.error("Error reading normalized CSV:", err.message);
    process.exit(1);
  }

  try {
    // Create windows from CSV
    const windows = makeWindowsFromNormalizedCsv(data, SEQ_LEN, STEP);
    console.log(`✅ Created ${windows.length} windows of length ${SEQ_LEN}`);

    // 5) Build a payload object to save as JSON on disk
    // You can later load this JSON in Python or JS
    const payload = {
      seq_len: SEQ_LEN,
      step: STEP,
      num_windows: windows.length,
      windows, // [num_windows][seq_len][3]
    };

    // Write JSON file to disk
    fs.writeFile(outputFile, JSON.stringify(payload), "utf8", err2 => {
      if (err2) {
        console.error("Error writing windows JSON:", err2.message);
        process.exit(1);
      }
      console.log("✅ Windows saved to:", outputFile);
    });
  } catch (e) {
    console.error("Error creating windows:", e.message);
    process.exit(1);
  }
});

// Export the main function so you can reuse it from other files if needed
module.exports = {
  makeWindowsFromNormalizedCsv,
};
