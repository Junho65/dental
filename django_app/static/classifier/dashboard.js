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
    const detectionCount = document.getElementById("detection-count");
    const topConfidence = document.getElementById("top-confidence");
    const summaryStrip = document.getElementById("summary-strip");
    const detectionList = document.getElementById("detection-list");

    const state = {
        file: null,
        objectUrl: null,
        detections: [],
    };

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

    const setStatus = (message, tone) => {
        statusBox.textContent = message;
        statusBox.classList.remove("is-busy", "is-error");
        if (tone === "busy") {
            statusBox.classList.add("is-busy");
        } else if (tone === "error") {
            statusBox.classList.add("is-error");
        }
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

    /** Wipe the overlay bitmap so boxes from a previous image never linger while the next image loads. */
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

        const scaleX = width / previewImage.naturalWidth;
        const scaleY = height / previewImage.naturalHeight;

        state.detections.forEach((detection, index) => {
            const info = classMap.get(detection.class_name) || {};
            const color = info.color || "#ffffff";
            const [x1, y1, x2, y2] = detection.bbox_xyxy;
            const left = x1 * scaleX;
            const top = y1 * scaleY;
            const boxWidth = (x2 - x1) * scaleX;
            const boxHeight = (y2 - y1) * scaleY;
            const label = `${index + 1}. ${detection.class_name} ${fmtPercent(detection.confidence)}`;

            ctx.lineWidth = 2;
            ctx.strokeStyle = color;
            ctx.fillStyle = `${color}22`;
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
    };

    const resizeCanvas = () => {
        if (!previewImage.complete || !previewImage.naturalWidth) {
            clearOverlayBitmap();
            return;
        }

        const rect = previewImage.getBoundingClientRect();
        const ratio = window.devicePixelRatio || 1;
        overlayCanvas.width = rect.width * ratio;
        overlayCanvas.height = rect.height * ratio;
        overlayCanvas.style.width = `${rect.width}px`;
        overlayCanvas.style.height = `${rect.height}px`;

        const ctx = overlayCanvas.getContext("2d");
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        drawDetections();
    };

    const renderList = () => {
        if (!state.detections.length) {
            summaryStrip.textContent = "아직 분석 결과가 없습니다.";
            detectionList.innerHTML =
                '<div class="placeholder-row">탐지 결과가 생기면 클래스, confidence, 좌표가 여기에 정리됩니다.</div>';
            detectionCount.textContent = "0";
            topConfidence.textContent = "-";
            drawDetections();
            return;
        }

        const sorted = [...state.detections].sort((a, b) => b.confidence - a.confidence);
        const grouped = sorted.reduce((acc, item) => {
            acc[item.class_name] = (acc[item.class_name] || 0) + 1;
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
                const [x1, y1, x2, y2] = item.bbox_xyxy.map((value) => value.toFixed(1));
                return `
                    <div class="detection-row">
                        <div class="detection-main">
                            <span class="result-dot" style="background:${info.color || "#ffffff"};"></span>
                            <div>
                                <strong>${index + 1}. ${info.label || item.class_name}</strong>
                                <div class="detection-meta">${info.description || ""}</div>
                                <div class="detection-meta">bbox: [${x1}, ${y1}, ${x2}, ${y2}]</div>
                            </div>
                        </div>
                        <div class="detection-score">${fmtPercent(item.confidence)}</div>
                    </div>
                `;
            })
            .join("");

        drawDetections();
    };

    const setFile = (file) => {
        if (!file) {
            return;
        }

        clearObjectUrl();
        state.file = file;
        state.objectUrl = URL.createObjectURL(file);
        state.detections = [];

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
        fileInput.value = "";
        previewImage.removeAttribute("src");
        fileName.textContent = "아직 선택되지 않음";
        fileSize.textContent = "-";
        updateViewerVisibility();
        renderList();
        setStatus("업로드 후 분석을 실행하면 탐지 결과가 오른쪽 뷰어와 아래 목록에 반영됩니다.");
    };

    const runPrediction = async () => {
        if (!state.file) {
            setStatus("먼저 X-ray 이미지를 선택하세요.", "error");
            return;
        }

        submitButton.disabled = true;
        setStatus("모델 추론을 실행 중입니다. 이미지 크기에 따라 수 초 정도 걸릴 수 있습니다.", "busy");

        const clientStarted = performance.now();

        try {
            const formData = new FormData();
            formData.append("image", state.file);

            const response = await fetch("/predict/", {
                method: "POST",
                body: formData,
                cache: "no-store",
            });
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
                timingNote = ` 서버 ${serverTotal}ms(모델 ${serverInfer}ms) · 왕복 ${roundTripMs}ms`;
            } else if (serverInfer != null) {
                timingNote = ` 서버 추론 ${serverInfer}ms · 왕복 ${roundTripMs}ms`;
            }

            state.detections = Array.isArray(payload.detections) ? payload.detections : [];
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
        } catch (error) {
            state.detections = [];
            renderList();
            setStatus(`분석 실패: ${error.message}`, "error");
        } finally {
            submitButton.disabled = false;
        }
    };

    previewImage.addEventListener("load", resizeCanvas);
    window.addEventListener("resize", resizeCanvas);

    // Reset so choosing another file (even with the same name) always fires `change`.
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

    updateViewerVisibility();
    renderList();
})();
