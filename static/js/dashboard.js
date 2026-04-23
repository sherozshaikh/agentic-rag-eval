(function () {
  "use strict";

  function readChartData() {
    var el = document.getElementById("dashboard-chart-data");
    if (!el) return null;
    try {
      return JSON.parse(el.textContent || "{}");
    } catch (err) {
      console.error("dashboard: failed to parse chart data", err);
      return null;
    }
  }

  function cssVar(name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name);
    return (v && v.trim()) || fallback;
  }

  var palette = [
    "#4d8bff", "#3fb950", "#e3b341", "#f85149", "#a371f7",
    "#2dd4bf", "#f97316", "#ec4899", "#22d3ee", "#facc15",
  ];

  function withAlpha(hex, alpha) {
    if (!hex || hex[0] !== "#") return hex;
    var h = hex.slice(1);
    if (h.length === 3) {
      h = h.split("").map(function (c) { return c + c; }).join("");
    }
    var r = parseInt(h.slice(0, 2), 16);
    var g = parseInt(h.slice(2, 4), 16);
    var b = parseInt(h.slice(4, 6), 16);
    return "rgba(" + r + "," + g + "," + b + "," + alpha + ")";
  }

  function baseOptions() {
    var textColor = cssVar("--text", "#1a1f2b");
    var gridColor = cssVar("--border", "#e3e6eb");
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: textColor } },
        tooltip: { mode: "index", intersect: false },
      },
      scales: {
        x: { ticks: { color: textColor }, grid: { color: gridColor } },
        y: {
          ticks: { color: textColor },
          grid: { color: gridColor },
          beginAtZero: true,
        },
      },
    };
  }

  function initPipelineQualityChart(data) {
    var canvas = document.getElementById("chart-pipeline-quality");
    if (!canvas || !data || !data.labels) return;
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: data.labels,
        datasets: [
          {
            label: "EM",
            data: data.em || [],
            backgroundColor: withAlpha(palette[0], 0.75),
            borderColor: palette[0],
            borderWidth: 1,
          },
          {
            label: "F1",
            data: data.f1 || [],
            backgroundColor: withAlpha(palette[1], 0.75),
            borderColor: palette[1],
            borderWidth: 1,
          },
        ],
      },
      options: Object.assign(baseOptions(), {
        plugins: Object.assign({}, baseOptions().plugins, {
          title: { display: true, text: "Quality metrics by pipeline",
                   color: cssVar("--text", "#1a1f2b") },
        }),
      }),
    });
  }

  function initFailureModesChart(data) {
    var canvas = document.getElementById("chart-failure-modes");
    if (!canvas || !data || !data.failure_labels) return;
    new Chart(canvas, {
      type: "doughnut",
      data: {
        labels: data.failure_labels,
        datasets: [
          {
            data: data.failure_counts || [],
            backgroundColor: palette.slice(0, data.failure_labels.length),
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "right",
                    labels: { color: cssVar("--text", "#1a1f2b") } },
          title: { display: true, text: "Failure mode distribution",
                   color: cssVar("--text", "#1a1f2b") },
        },
      },
    });
  }

  function initRunMetricsChart(data) {
    var canvas = document.getElementById("chart-run-metrics");
    if (!canvas || !data || !data.labels) return;
    new Chart(canvas, {
      type: "radar",
      data: {
        labels: data.labels,
        datasets: [
          {
            label: "Run metrics",
            data: data.values || [],
            borderColor: palette[0],
            backgroundColor: withAlpha(palette[0], 0.25),
            pointBackgroundColor: palette[0],
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: cssVar("--text", "#1a1f2b") } },
        },
        scales: {
          r: {
            suggestedMin: 0,
            suggestedMax: 1,
            ticks: { color: cssVar("--text-muted", "#5b6472") },
            grid: { color: cssVar("--border", "#e3e6eb") },
            angleLines: { color: cssVar("--border", "#e3e6eb") },
            pointLabels: { color: cssVar("--text", "#1a1f2b") },
          },
        },
      },
    });
  }

  function initRunFailuresChart(data) {
    var canvas = document.getElementById("chart-run-failures");
    if (!canvas || !data || !data.failure_labels) return;
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: data.failure_labels,
        datasets: [
          {
            label: "Count",
            data: data.failure_counts || [],
            backgroundColor: withAlpha(palette[3], 0.75),
            borderColor: palette[3],
            borderWidth: 1,
          },
        ],
      },
      options: Object.assign(baseOptions(), {
        plugins: Object.assign({}, baseOptions().plugins, {
          title: { display: true, text: "Failure modes in this run",
                   color: cssVar("--text", "#1a1f2b") },
          legend: { display: false },
        }),
      }),
    });
  }

  function init() {
    if (typeof Chart === "undefined") {
      console.warn("dashboard: Chart.js not loaded");
      return;
    }
    var data = readChartData();
    if (!data) return;
    initPipelineQualityChart(data);
    initFailureModesChart(data);
    initRunMetricsChart(data);
    initRunFailuresChart(data);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
