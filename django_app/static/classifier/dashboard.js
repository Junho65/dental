(function () {
    const classInfo = JSON.parse(document.getElementById("class-info-data").textContent);
    const classMap = new Map(classInfo.map((item) => [item.name, item]));

    const form = document.getElementById("predict-form");
    const fileInput = document.getElementById("image-input");
    const dropzone = document.getElementById("dropzone");
    const resetButton = document.getElementById("reset-button");
    const submitButton = document.getElementById("submit-button");
    const fileName = document.getElementById("file-name");
    const fileSize = document.getElementById("file-size");
    const statusBox = document.getElementById("status-box");
    const viewerEmpty = document.getElementById("viewer-empty");
    const viewerMedia = document.getElementById("viewer-media");
    const previewImage = document.getElementById("preview-image");
    const overlayCanvas = document.getElementById("overlay-canvas");
    const revealHandle = document.getElementById("reveal-handle");
    const revealControls = document.getElementById("reveal-controls");
    const revealSlider = document.getElementById("reveal-slider");
    const detectionCount = document.getElementById("detection-count");
    const topConfidence = document.getElementById("top-confidence");
    const summaryStrip = document.getElementById("summary-strip");
    const detectionList = document.getElementById("detection-list");
    const treatmentSummary = document.getElementById("treatment-summary");
    const treatmentTableBody = document.getElementById("treatment-table-body");
    const treatmentTotal = document.getElementById("treatment-total");
    const statusMessage = document.getElementById("status-message");
    const statusStages = document.getElementById("status-stages");
    const classLegend = document.getElementById("class-legend");

    const STAGE_ORDER = ["upload", "detect", "refine", "render"];
    const LEGEND_CLASS_NAMES = [
        "caries",
        "deep_caries",
        "caries_family",
        "periapical_lesion",
        "impacted_tooth",
    ];

    const state = {
        file: null,
        objectUrl: null,
        detections: [],
        sortedDetections: [],
        activeIndex: null,
        revealRatio: 1,
    };

    const currencyFormatter = new Intl.NumberFormat("ko-KR");

    const escapeHtml = (value) =>
        String(value ?? "").replace(/[&<>"']/g, (char) => {
            const entities = {
                "&": "&amp;",
                "<": "&lt;",
                ">": "&gt;",
                '"': "&quot;",
                "'": "&#39;",
            };
            return entities[char] || char;
        });

    const fmtPercent = (value) => `${(value * 100).toFixed(1)}%`;

    const fmtSize = (bytes) => {
        if (!Number.isFinite(bytes)) {
            return "-";
        }
        if (bytes < 1024 * 1024) {
            return `${(bytes / 1024).toFixed(1)} KB`;
        }
        return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    };

    const fmtCurrency = (value) => {
        if (!Number.isFinite(value)) {
            return "-";
        }
        return `${currencyFormatter.format(Math.round(value))}원`;
    };

    const fmtRange = (min, max) => {
        const hasMin = Number.isFinite(min);
        const hasMax = Number.isFinite(max);
        if (!hasMin && !hasMax) {
            return "상담 필요";
        }
        if (hasMin && hasMax) {
            if (min === max) {
                return fmtCurrency(min);
            }
            return `${fmtCurrency(min)} ~ ${fmtCurrency(max)}`;
        }
        return fmtCurrency(hasMin ? min : max);
    };

    const getDisplayName = (className) => {
        const info = classMap.get(className) || {};
        return info.label_ko || info.label || className;
    };

    const getAreaLabel = (bbox) => {
        if (!Array.isArray(bbox) || bbox.length !== 4) {
            return "위치 추정";
        }

        const [x1, y1, x2, y2] = bbox;
        const sourceWidth = previewImage.naturalWidth || Math.max(x2, 1);
        const sourceHeight = previewImage.naturalHeight || Math.max(y2, 1);
        const centerX = ((x1 + x2) / 2) / sourceWidth;
        const centerY = ((y1 + y2) / 2) / sourceHeight;

        const vertical = centerY < 0.33 ? "상단" : centerY > 0.66 ? "하단" : "중앙";
        const horizontal = centerX < 0.33 ? "좌측" : centerX > 0.66 ? "우측" : "중앙";

        if (vertical === "중앙" && horizontal === "중앙") {
            return "중앙";
        }
        if (vertical === "중앙") {
            return horizontal;
        }
        if (horizontal === "중앙") {
            return vertical;
        }
        return `${vertical} ${horizontal}`;
    };

    const setStatus = (message, tone) => {
        if (statusMessage) {
            statusMessage.textContent = message;
        } else {
            statusBox.textContent = message;
        }
        statusBox.classList.remove("is-busy", "is-error");
        if (tone === "busy") {
            statusBox.classList.add("is-busy");
        } else if (tone === "error") {
            statusBox.classList.add("is-error");
        }
    };

    const setStages = (currentStage) => {
        if (!statusStages) {
            return;
        }
        if (!currentStage) {
            statusStages.classList.add("is-hidden");
            statusStages.querySelectorAll(".status-stage").forEach((node) => {
                node.classList.remove("is-active", "is-done");
            });
            return;
        }
        statusStages.classList.remove("is-hidden");
        const currentIdx = STAGE_ORDER.indexOf(currentStage);
        statusStages.querySelectorAll(".status-stage").forEach((node) => {
            const stage = node.dataset.stage;
            const idx = STAGE_ORDER.indexOf(stage);
            node.classList.toggle("is-done", idx >= 0 && currentIdx > idx);
            node.classList.toggle("is-active", idx === currentIdx);
        });
    };

    const renderLegend = () => {
        if (!classLegend) {
            return;
        }
        classLegend.innerHTML = LEGEND_CLASS_NAMES.map((name) => {
            const info = classMap.get(name);
            if (!info) {
                return "";
            }
            return `
                <li class="legend-item">
                    <span class="legend-swatch" style="background:${info.color || "#ffffff"};"></span>
                    <span>${escapeHtml(info.label_ko || info.label || name)}</span>
                </li>
            `;
        }).join("");
    };

    const updateViewerVisibility = () => {
        const hasImage = Boolean(state.objectUrl);
        viewerEmpty.style.display = hasImage ? "none" : "block";
        viewerMedia.classList.toggle("is-hidden", !hasImage);
    };

    const clearObjectUrl = () => {
        if (state.objectUrl) {
            URL.revokeObjectURL(state.objectUrl);
            state.objectUrl = null;
        }
    };

    const clearOverlayBitmap = () => {
        const ctx = overlayCanvas.getContext("2d");
        if (!ctx) {
            return;
        }
        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        ctx.restore();
    };

    const drawDetections = () => {
        const ctx = overlayCanvas.getContext("2d");
        if (!ctx) {
            return;
        }

        clearOverlayBitmap();

        if (!previewImage.naturalWidth) {
            return;
        }

        const rect = previewImage.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        const revealX = width * state.revealRatio;

        if (revealX <= 0) {
            return;
        }

        const scaleX = width / previewImage.naturalWidth;
        const scaleY = height / previewImage.naturalHeight;
        const hasActive = state.activeIndex != null;
        const renderList = state.sortedDetections.length
            ? state.sortedDetections
            : state.detections;

        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, revealX, height);
        ctx.clip();

        renderList.forEach((detection, index) => {
            const info = classMap.get(detection.class_name) || {};
            const color = info.color || "#ffffff";
            const [x1, y1, x2, y2] = detection.bbox_xyxy;
            const left = x1 * scaleX;
            const top = y1 * scaleY;
            const boxWidth = (x2 - x1) * scaleX;
            const boxHeight = (y2 - y1) * scaleY;
            const isActive = hasActive && index === state.activeIndex;
            const dim = hasActive && !isActive;
            const label = `${index + 1}. ${getDisplayName(detection.class_name)} ${fmtPercent(detection.confidence)}`;

            ctx.globalAlpha = dim ? 0.32 : 1;
            ctx.lineWidth = isActive ? 3.5 : 2;
            ctx.strokeStyle = color;
            ctx.fillStyle = `${color}${isActive ? "3c" : "22"}`;
            ctx.strokeRect(left, top, boxWidth, boxHeight);
            ctx.fillRect(left, top, boxWidth, boxHeight);

            ctx.font = "600 12px Aptos, Bahnschrift, sans-serif";
            const textWidth = ctx.measureText(label).width;
            const labelY = Math.max(22, top);
            ctx.fillStyle = color;
            ctx.fillRect(left, labelY - 20, textWidth + 16, 20);
            ctx.fillStyle = "#0e1417";
            ctx.fillText(label, left + 8, labelY - 6);
        });

        ctx.globalAlpha = 1;
        ctx.restore();
    };

    const updateRevealHandle = () => {
        if (!revealHandle) {
            return;
        }
        const hasImage = Boolean(state.objectUrl) && state.detections.length > 0;
        revealHandle.classList.toggle("is-visible", hasImage && state.revealRatio < 1);
        if (!hasImage) {
            return;
        }
        const rect = previewImage.getBoundingClientRect();
        const mediaRect = viewerMedia.getBoundingClientRect();
        const offsetLeft = rect.left - mediaRect.left;
        const x = offsetLeft + rect.width * state.revealRatio;
        revealHandle.style.left = `${x}px`;
    };

    const updateRevealControlsVisibility = () => {
        if (!revealControls) {
            return;
        }
        const show = Boolean(state.objectUrl) && state.detections.length > 0;
        revealControls.classList.toggle("is-hidden", !show);
    };

    const hitTestDetection = (clientX, clientY) => {
        if (!previewImage.naturalWidth) {
            return null;
        }
        const rect = previewImage.getBoundingClientRect();
        const x = clientX - rect.left;
        const y = clientY - rect.top;
        if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
            return null;
        }
        const scaleX = rect.width / previewImage.naturalWidth;
        const scaleY = rect.height / previewImage.naturalHeight;
        const list = state.sortedDetections.length ? state.sortedDetections : state.detections;
        let bestIndex = null;
        let bestArea = Infinity;
        list.forEach((detection, index) => {
            const [x1, y1, x2, y2] = detection.bbox_xyxy;
            const left = x1 * scaleX;
            const top = y1 * scaleY;
            const right = x2 * scaleX;
            const bottom = y2 * scaleY;
            if (x >= left && x <= right && y >= top && y <= bottom) {
                const area = (right - left) * (bottom - top);
                if (area < bestArea) {
                    bestArea = area;
                    bestIndex = index;
                }
            }
        });
        return bestIndex;
    };

    const updateActiveHighlight = () => {
        const rows = detectionList.querySelectorAll(".detection-row");
        rows.forEach((row) => {
            const idx = Number(row.dataset.index);
            row.classList.toggle("is-active", state.activeIndex != null && idx === state.activeIndex);
        });
        detectionList.classList.toggle("has-active", state.activeIndex != null);
    };

    const setActiveDetection = (index, options) => {
        const opts = options || {};
        const next = index == null || index < 0 ? null : index;
        const changed = state.activeIndex !== next;
        state.activeIndex = next;
        updateActiveHighlight();
        drawDetections();
        if (changed && next != null && opts.scrollIntoView) {
            const row = detectionList.querySelector(`.detection-row[data-index="${next}"]`);
            if (row) {
                row.scrollIntoView({ behavior: "smooth", block: "nearest" });
            }
        }
    };

    const resizeCanvas = () => {
        if (!previewImage.complete || !previewImage.naturalWidth) {
            clearOverlayBitmap();
            updateRevealHandle();
            return;
        }

        const rect = previewImage.getBoundingClientRect();
        const mediaRect = viewerMedia.getBoundingClientRect();
        const offsetLeft = rect.left - mediaRect.left;
        const offsetTop = rect.top - mediaRect.top;
        const ratio = window.devicePixelRatio || 1;
        overlayCanvas.width = rect.width * ratio;
        overlayCanvas.height = rect.height * ratio;
        overlayCanvas.style.width = `${rect.width}px`;
        overlayCanvas.style.height = `${rect.height}px`;
        overlayCanvas.style.left = `${offsetLeft}px`;
        overlayCanvas.style.top = `${offsetTop}px`;
        overlayCanvas.style.right = "auto";
        overlayCanvas.style.bottom = "auto";

        const ctx = overlayCanvas.getContext("2d");
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        drawDetections();
        updateRevealHandle();
    };

    const renderTreatmentTable = (detections) => {
        if (!detections.length) {
            treatmentSummary.textContent = "아직 분석 결과가 없습니다.";
            treatmentTableBody.innerHTML =
                '<tr class="cost-placeholder"><td colspan="4">분석 결과가 생기면 부위별 병명과 예상 치료비가 여기에 정리됩니다.</td></tr>';
            treatmentTotal.textContent = "-";
            return;
        }

        let totalMin = 0;
        let totalMax = 0;

        const grouped = detections.reduce((acc, item) => {
            const key = getDisplayName(item.class_name);
            acc[key] = (acc[key] || 0) + 1;
            return acc;
        }, {});
        treatmentSummary.textContent = Object.entries(grouped)
            .map(([name, count]) => `${name} ${count}건`)
            .join(" / ");

        treatmentTableBody.innerHTML = detections
            .map((item, index) => {
                const info = classMap.get(item.class_name) || {};
                const costMin = Number(info.cost_min);
                const costMax = Number(info.cost_max);
                if (Number.isFinite(costMin)) {
                    totalMin += costMin;
                }
                if (Number.isFinite(costMax)) {
                    totalMax += costMax;
                }

                return `
                    <tr>
                        <td>
                            <strong>#${index + 1} ${escapeHtml(getAreaLabel(item.bbox_xyxy))}</strong>
                            <div class="cost-cell-sub">bbox 중심 좌표 기준</div>
                        </td>
                        <td>
                            <strong>${escapeHtml(getDisplayName(item.class_name))}</strong>
                            <div class="cost-cell-sub">${escapeHtml(fmtPercent(item.confidence))} 신뢰도</div>
                        </td>
                        <td>${escapeHtml(info.treatment || "치과 전문의 상담 필요")}</td>
                        <td class="cost-amount">${escapeHtml(fmtRange(costMin, costMax))}</td>
                    </tr>
                `;
            })
            .join("");

        treatmentTotal.textContent = fmtRange(totalMin, totalMax);
    };

    const buildSeverityChip = (detection) => {
        const sevConf = Number(detection.severity_confidence);
        if (!Number.isFinite(sevConf)) {
            return "";
        }
        const className = detection.class_name;
        const probs = detection.severity_probabilities || {};
        let label;
        let chipClass;
        if (className === "deep_caries") {
            label = `심부 충치 의심 ${fmtPercent(sevConf)}`;
            chipClass = "is-severe";
        } else if (className === "caries") {
            label = `초기/중기 충치 의심 ${fmtPercent(sevConf)}`;
            chipClass = "is-mild";
        } else {
            const deepProb = Number(probs.deep_caries);
            if (Number.isFinite(deepProb) && deepProb > 0.5) {
                label = `심부 충치 가능성 ${fmtPercent(deepProb)}`;
                chipClass = "is-severe";
            } else {
                label = `세부 분류 보류 (신뢰도 ${fmtPercent(sevConf)})`;
                chipClass = "is-uncertain";
            }
        }
        return `<span class="severity-chip ${chipClass}">${escapeHtml(label)}</span>`;
    };

    const renderList = () => {
        if (!state.detections.length) {
            summaryStrip.textContent = "아직 분석 결과가 없습니다.";
            detectionList.innerHTML =
                '<div class="placeholder-row">탐지 결과가 생기면 클래스와 신뢰도가 여기에 정리됩니다.</div>';
            detectionList.classList.remove("has-active");
            detectionCount.textContent = "0";
            topConfidence.textContent = "-";
            state.sortedDetections = [];
            state.activeIndex = null;
            renderTreatmentTable([]);
            updateRevealControlsVisibility();
            updateRevealHandle();
            drawDetections();
            return;
        }

        const sorted = [...state.detections].sort((a, b) => b.confidence - a.confidence);
        state.sortedDetections = sorted;
        if (state.activeIndex != null && state.activeIndex >= sorted.length) {
            state.activeIndex = null;
        }
        const grouped = sorted.reduce((acc, item) => {
            const key = getDisplayName(item.class_name);
            acc[key] = (acc[key] || 0) + 1;
            return acc;
        }, {});

        detectionCount.textContent = String(sorted.length);
        topConfidence.textContent = fmtPercent(sorted[0].confidence);
        summaryStrip.textContent = Object.entries(grouped)
            .map(([name, count]) => `${name} ${count}건`)
            .join(" / ");

        detectionList.innerHTML = sorted
            .map((item, index) => {
                const info = classMap.get(item.class_name) || {};
                const severityChip = buildSeverityChip(item);
                const explanation = info.patient_explanation
                    ? `<div class="patient-explanation">${escapeHtml(info.patient_explanation)}</div>`
                    : "";
                const nextStep = info.patient_next_step
                    ? `<div class="patient-next-step">다음 단계: ${escapeHtml(info.patient_next_step)}</div>`
                    : "";
                const badges = severityChip
                    ? `<div class="detection-badges">${severityChip}</div>`
                    : "";
                return `
                    <div class="detection-row" data-index="${index}" role="button" tabindex="0">
                        <div class="detection-main">
                            <span class="result-dot" style="background:${info.color || "#ffffff"};"></span>
                            <div>
                                <strong>${index + 1}. ${escapeHtml(getDisplayName(item.class_name))}</strong>
                                <div class="detection-meta">부위: ${escapeHtml(getAreaLabel(item.bbox_xyxy))}</div>
                                ${badges}
                                ${explanation}
                                ${nextStep}
                            </div>
                        </div>
                        <div class="detection-score">${escapeHtml(fmtPercent(item.confidence))}</div>
                    </div>
                `;
            })
            .join("");

        updateActiveHighlight();
        renderTreatmentTable(sorted);
        updateRevealControlsVisibility();
        updateRevealHandle();
        drawDetections();
    };

    const resetRevealToFull = () => {
        state.revealRatio = 1;
        if (revealSlider) {
            revealSlider.value = "100";
        }
    };

    const setFile = (file) => {
        if (!file) {
            return;
        }

        clearObjectUrl();
        state.file = file;
        state.objectUrl = URL.createObjectURL(file);
        state.detections = [];
        state.sortedDetections = [];
        state.activeIndex = null;
        resetRevealToFull();

        fileName.textContent = file.name;
        fileSize.textContent = fmtSize(file.size);
        previewImage.src = state.objectUrl;
        updateViewerVisibility();
        renderList();
        setStatus("이미지 프리뷰가 준비되었습니다. 이제 AI 분석을 실행할 수 있습니다.");
    };

    const resetAll = () => {
        clearObjectUrl();
        state.file = null;
        state.detections = [];
        state.sortedDetections = [];
        state.activeIndex = null;
        resetRevealToFull();
        fileInput.value = "";
        previewImage.removeAttribute("src");
        fileName.textContent = "아직 선택하지 않음";
        fileSize.textContent = "-";
        updateViewerVisibility();
        renderList();
        setStages(null);
        setStatus("업로드 후 분석을 실행하면 탐지 결과가 오른쪽 뷰어와 아래 목록에 반영됩니다.");
    };

    const runPrediction = async () => {
        if (!state.file) {
            setStatus("먼저 X-ray 이미지를 선택하세요.", "error");
            return;
        }

        submitButton.disabled = true;
        setStatus("이미지를 서버로 보내고 있습니다.", "busy");
        setStages("upload");

        const clientStarted = performance.now();

        try {
            const formData = new FormData();
            formData.append("image", state.file);

            const fetchPromise = fetch("/predict/", {
                method: "POST",
                body: formData,
                cache: "no-store",
            });

            window.setTimeout(() => {
                if (submitButton.disabled) {
                    setStatus("AI 모델이 X-ray에서 의심 부위를 찾고 있습니다.", "busy");
                    setStages("detect");
                }
            }, 600);

            const response = await fetchPromise;
            setStatus("탐지된 부위의 심각도를 분석 중입니다.", "busy");
            setStages("refine");
            const contentType = response.headers.get("content-type") || "";
            const rawBody = await response.text();
            const payload = contentType.includes("application/json")
                ? JSON.parse(rawBody)
                : { error: rawBody.slice(0, 240) || "Non-JSON response returned" };

            if (!response.ok) {
                throw new Error(payload.error || "Prediction request failed");
            }

            const roundTripMs = Math.round(performance.now() - clientStarted);
            const serverInfer =
                typeof payload.inference_ms === "number" && Number.isFinite(payload.inference_ms)
                    ? payload.inference_ms
                    : null;
            const serverTotal =
                typeof payload.server_total_ms === "number" && Number.isFinite(payload.server_total_ms)
                    ? payload.server_total_ms
                    : null;
            let timingNote = ` 왕복 ${roundTripMs}ms`;
            if (serverTotal != null && serverInfer != null) {
                timingNote = ` 서버 ${serverTotal}ms(모델 ${serverInfer}ms) / 왕복 ${roundTripMs}ms`;
            } else if (serverInfer != null) {
                timingNote = ` 서버 추론 ${serverInfer}ms / 왕복 ${roundTripMs}ms`;
            }

            state.detections = Array.isArray(payload.detections) ? payload.detections : [];
            state.activeIndex = null;
            resetRevealToFull();
            setStages("render");
            renderList();

            if (state.detections.length) {
                setStatus(
                    `분석이 완료되었습니다. ${state.detections.length}개의 의심 부위를 시각화했습니다.${timingNote}`,
                );
            } else {
                setStatus(
                    `분석이 완료되었습니다. 현재 기준에서는 탐지된 의심 부위가 없습니다.${timingNote}`,
                );
            }
            window.setTimeout(() => setStages(null), 1200);
        } catch (error) {
            state.detections = [];
            renderList();
            setStatus(`분석 실패: ${error.message}`, "error");
            setStages(null);
        } finally {
            submitButton.disabled = false;
        }
    };

    previewImage.addEventListener("load", () => {
        resizeCanvas();
        renderList();
    });
    window.addEventListener("resize", resizeCanvas);

    if (revealSlider) {
        revealSlider.addEventListener("input", (event) => {
            const raw = Number(event.target.value);
            state.revealRatio = Number.isFinite(raw) ? Math.max(0, Math.min(1, raw / 100)) : 1;
            drawDetections();
            updateRevealHandle();
        });
    }

    overlayCanvas.addEventListener("click", (event) => {
        const idx = hitTestDetection(event.clientX, event.clientY);
        if (idx == null) {
            setActiveDetection(null);
        } else {
            setActiveDetection(idx, { scrollIntoView: true });
        }
    });

    detectionList.addEventListener("click", (event) => {
        const row = event.target.closest(".detection-row");
        if (!row || !row.dataset.index) {
            return;
        }
        const idx = Number(row.dataset.index);
        const next = state.activeIndex === idx ? null : idx;
        setActiveDetection(next);
    });

    detectionList.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") {
            return;
        }
        const row = event.target.closest(".detection-row");
        if (!row || !row.dataset.index) {
            return;
        }
        event.preventDefault();
        const idx = Number(row.dataset.index);
        const next = state.activeIndex === idx ? null : idx;
        setActiveDetection(next);
    });

    fileInput.addEventListener("click", () => {
        fileInput.value = "";
    });

    fileInput.addEventListener("change", (event) => {
        setFile(event.target.files && event.target.files[0]);
    });

    ["dragenter", "dragover"].forEach((type) => {
        dropzone.addEventListener(type, (event) => {
            event.preventDefault();
            dropzone.classList.add("is-dragover");
        });
    });

    ["dragleave", "dragend", "drop"].forEach((type) => {
        dropzone.addEventListener(type, (event) => {
            event.preventDefault();
            dropzone.classList.remove("is-dragover");
        });
    });

    dropzone.addEventListener("drop", (event) => {
        const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
        if (file) {
            setFile(file);
        }
    });

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        runPrediction();
    });

    resetButton.addEventListener("click", resetAll);

    renderLegend();
    updateViewerVisibility();
    renderList();
})();
