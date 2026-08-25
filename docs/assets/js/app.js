(function () {
  const data = window.DEMO_DATA;
  const sampleList = document.querySelector("#sample-list");
  const datasetSummary = document.querySelector("#dataset-summary");
  const tabs = Array.from(document.querySelectorAll(".dataset-tab"));

  let activeDataset = "clean";

  function createElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) {
      element.className = className;
    }
    if (text) {
      element.textContent = text;
    }
    return element;
  }

  function audioPath(datasetKey, sampleId, methodKey) {
    const dataset = data.datasets[datasetKey];
    return `assets/audio/${dataset.folder}/${sampleId}/${methodKey}.wav`;
  }

  function renderSummary(datasetKey) {
    const dataset = data.datasets[datasetKey];
    const summaryItems = [
      ["Dataset", dataset.label],
      ["Condition", dataset.summary.condition],
      ["Samples", dataset.summary.samples],
      ["Bitrate range", dataset.summary.bitrate],
    ];

    datasetSummary.replaceChildren();
    summaryItems.forEach(([label, value]) => {
      const chip = createElement("div", "summary-chip");
      chip.append(createElement("span", "", label));
      chip.append(createElement("strong", "", value));
      datasetSummary.append(chip);
    });
  }

  function renderSamples(datasetKey) {
    const dataset = data.datasets[datasetKey];
    sampleList.replaceChildren();

    dataset.samples.forEach((sample) => {
      const card = createElement("article", "sample-card");

      const header = createElement("div", "sample-header");
      const copy = createElement("div");
      copy.append(createElement("h3", "sample-title", sample.title));
      copy.append(createElement("p", "sample-transcript", sample.transcript));

      const meta = createElement("div", "sample-meta");
      meta.append(createElement("span", "", dataset.label));
      meta.append(createElement("span", "", sample.speaker));
      meta.append(createElement("span", "", sample.id));

      header.append(copy, meta);
      card.append(header);

      const table = createElement("div", "audio-table");
      data.methods.forEach((method) => {
        const row = createElement("div", `audio-row${method.ours ? " ours" : ""}`);
        const methodInfo = createElement("div", "method-info");
        const methodName = createElement("div", "method-name", method.label);
        if (method.ours) {
          methodName.append(createElement("span", "method-pill", "ours"));
        }
        methodInfo.append(methodName);
        methodInfo.append(createElement("div", "method-meta", method.details));

        const control = createElement("div", "audio-control");
        const audio = document.createElement("audio");
        audio.controls = true;
        audio.preload = "none";
        audio.src = audioPath(datasetKey, sample.id, method.key);

        const state = createElement("span", "file-state", "Audio pending");
        state.dataset.audioPath = audio.src;
        control.append(audio, state);
        row.append(methodInfo, control);
        table.append(row);
      });

      card.append(table);
      sampleList.append(card);
    });

    checkAudioFiles();
  }

  function checkAudioFiles() {
    if (!window.location.protocol.startsWith("http")) {
      return;
    }

    document.querySelectorAll(".file-state").forEach(async (state) => {
      try {
        const response = await fetch(state.dataset.audioPath, { method: "HEAD" });
        if (response.ok) {
          state.classList.add("is-ready");
          state.textContent = "Ready";
        }
      } catch (error) {
        state.textContent = "Audio pending";
      }
    });
  }

  function setDataset(datasetKey) {
    activeDataset = datasetKey;
    tabs.forEach((tab) => {
      const selected = tab.dataset.dataset === activeDataset;
      tab.classList.toggle("is-active", selected);
      tab.setAttribute("aria-selected", String(selected));
    });
    renderSummary(activeDataset);
    renderSamples(activeDataset);
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => setDataset(tab.dataset.dataset));
  });

  setDataset(activeDataset);
})();
