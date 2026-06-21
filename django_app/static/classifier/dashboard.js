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
    const detectionCount = document.getElementById("detection-count");
    const summaryStrip = document.getElementById("summary-strip");
    const detectionList = document.getElementById("detection-list");
    const treatmentSummary = document.getElementById("treatment-summary");
    const treatmentTableBody = document.getElementById("treatment-table-body");
    const treatmentTotal = document.getElementById("treatment-total");
    const statusMessage = document.getElementById("status-message");
    const statusStages = document.getElementById("status-stages");
    const classLegend = document.getElementById("class-legend");
    const insightGrid = document.querySelector(".insight-grid");
    const treatmentSection = document.querySelector(".treatment-section");
    const resultsDisclaimer = document.querySelector(".results-disclaimer");

    const STAGE_ORDER = ["upload", "detect", "refine", "render"];

    const showResultSections = () => {
        [insightGrid, treatmentSection, resultsDisclaimer].forEach((el) => {
            if (!el) return;
            el.classList.remove("is-hidden");
        });
    };

    const hideResultSections = () => {
        [insightGrid, treatmentSection, resultsDisclaimer].forEach((el) => {
            if (!el) return;
            el.classList.add("is-hidden");
        });
    };
    const LEGEND_CLASS_NAMES = classInfo.map((item) => item.name);
    const PERIODONTAL_CLASS_NAMES = new Set(["bone_loss", "furcation_involvement"]);
    const PERIODONTAL_STAGE_LABELS = {
        mild: "경도",
        medium: "중등도",
        severe: "중증",
    };

    const state = {
        file: null,
        objectUrl: null,
        detections: [],
        sortedDetections: [],
        activeIndex: null,
        revealRatio: 1,
        hospitalFilter: "clinic",
        surgeryFilter: "primary",
        disabilityFilter: "general",
        treatmentScopeFilter: "all",
        expandedLesionCards: {},
    };

    const FALLBACK_HOSPITAL_PATTERNS = {
        clinic: /(의원|치과의원|보건의료원)/,
        general: /(종합|상급종합|종병이상|치과병원)/,
        dental_univ: /(치대부속|치과대학|대학부속)/,
    };

    const getStructuredHospitalCategory = (option) => {
        if (option?.hospital_category) {
            return option.hospital_category;
        }
        const fullText = [option?.hospital_label, option?.name, option?.full_name]
            .filter(Boolean)
            .join(" ");
        if (FALLBACK_HOSPITAL_PATTERNS.dental_univ.test(fullText)) {
            return "dental_univ";
        }
        if (FALLBACK_HOSPITAL_PATTERNS.general.test(fullText)) {
            return "general";
        }
        if (FALLBACK_HOSPITAL_PATTERNS.clinic.test(fullText)) {
            return "clinic";
        }
        return "all";
    };

    const getStructuredSurgeryRole = (option) => {
        if (option?.surgery_role) {
            return option.surgery_role;
        }
        const fullText = [option?.name, option?.full_name]
            .filter(Boolean)
            .join(" ");
        return /제2의수술/.test(fullText) ? "secondary" : "primary";
    };

    const isStructuredDisabledOption = (option) => {
        if (typeof option?.disability_surcharge === "boolean") {
            return option.disability_surcharge;
        }
        const fullText = [option?.name, option?.full_name, option?.label]
            .filter(Boolean)
            .join(" ");
        return /장애인|장애 가산|장애가산|장애/.test(fullText);
    };

    const matchesStructuredHospitalFilter = (option) => {
        if (state.hospitalFilter === "all") {
            return true;
        }
        return getStructuredHospitalCategory(option) === state.hospitalFilter;
    };

    const matchesStructuredSurgeryFilter = (option) => {
        const surgeryRole = getStructuredSurgeryRole(option);
        switch (state.surgeryFilter) {
            case "primary":
                return surgeryRole !== "secondary";
            case "secondary":
                return surgeryRole === "secondary";
            default:
                return true;
        }
    };

    const HOSPITAL_FILTER_HINTS = {
        all: "모든 의료기관의 예상 비용을 표시합니다.",
        clinic: "치과의원(동네 치과) 기준입니다. 방문 예정 병원이 치과의원이라면 이 요금이 적용됩니다.",
        general: "종합병원·상급종합병원 기준입니다. 대형 병원 치과나 대학병원 치과(비치대부속)가 해당됩니다.",
        dental_univ: "치과대학부속병원 기준입니다. 서울대, 연세대, 경희대 등 치과대학에 딸린 병원이 해당됩니다.",
    };

    const SURGERY_FILTER_HINTS = {
        all: "",
        primary: " 이 시술이 해당 방문의 유일하거나 주된 수술인 경우입니다.",
        secondary: " 같은 방문에서 다른 수술도 함께 시행할 때의 요금입니다(단독 수술보다 저렴하게 책정됩니다).",
    };

    const DISABILITY_FILTER_HINTS = {
        general: " 장애인 가산이 명시되지 않은 옵션만 표시합니다.",
        disabled: " 장애인 가산이 명시된 옵션만 표시합니다. 현재 저장된 공공 수가에 해당 항목이 없으면 빈 결과가 나올 수 있습니다.",
    };

    const TREATMENT_SCOPE_HINTS = {
        all: " 대표 진료항목과 추가 진료항목을 함께 표시합니다.",
        primary_only: " 대표 진료항목만 표시합니다.",
    };

    const isCostFilterActive = () =>
        state.hospitalFilter !== "all" || state.surgeryFilter !== "all" || state.disabilityFilter !== "general";

    const updateFilterHint = () => {
        if (!filterHint) {
            return;
        }
        const hintH = HOSPITAL_FILTER_HINTS[state.hospitalFilter] || "";
        const hintS = SURGERY_FILTER_HINTS[state.surgeryFilter] || "";
        const hintD = DISABILITY_FILTER_HINTS[state.disabilityFilter] || "";
        const hintT = TREATMENT_SCOPE_HINTS[state.treatmentScopeFilter] || "";
        filterHint.textContent =
            hintH + hintS + hintD + hintT || "장애인 가산이 명시되지 않은 일반 옵션만 표시합니다.";
    };

    const getFilteredOptions = (options) => {
        if (!isCostFilterActive()) {
            return options;
        }
        const filtered = options.filter(
            (opt) =>
                matchesStructuredHospitalFilter(opt) &&
                matchesStructuredSurgeryFilter(opt) &&
                (state.disabilityFilter === "general"
                    ? !isStructuredDisabledOption(opt)
                    : state.disabilityFilter === "disabled"
                        ? isStructuredDisabledOption(opt)
                        : true)
        );
        return filtered;
    };

    const currencyFormatter = new Intl.NumberFormat("ko-KR");

    const buildOptionMeta = (option) => {
        const parts = [];
        if (option?.hospital_label) {
            parts.push(option.hospital_label);
        }
        if (getStructuredSurgeryRole(option) === "secondary") {
            parts.push("제2의수술");
        }
        if (isStructuredDisabledOption(option)) {
            parts.push("장애인 가산");
        }
        return parts.join(" · ");
    };

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

    const toFiniteNumber = (value) => {
        if (value == null || value === "") {
            return NaN;
        }
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : NaN;
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

    const getDisplayInfo = (className) => {
        return classMap.get(className) || {};
    };

    const getDisplayName = (className) => {
        const info = getDisplayInfo(className);
        return info.label_ko || info.label || className;
    };

    const getOfficialName = (className) => {
        const info = getDisplayInfo(className);
        return info.official_label_ko || info.label_ko || info.label || className;
    };

    const getTreatmentDisplayName = (className) => {
        if (className === "caries") {
            return "충치";
        }
        return getDisplayName(className);
    };

    const getTreatmentOfficialName = (className) => {
        if (className === "caries") {
            return "치아우식증";
        }
        return getOfficialName(className);
    };

    const getClassColor = (className) => {
        const info = getDisplayInfo(className);
        return info.color || "#ffffff";
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
            const info = getDisplayInfo(name);
            if (!info) {
                return "";
            }
            const officialName = getOfficialName(name);
            return `
                <li class="legend-item">
                    <span class="legend-swatch" style="background:${getClassColor(name)};"></span>
                    <span class="legend-copy">
                        <strong>${escapeHtml(officialName)}</strong>
                    </span>
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
            const color = getClassColor(detection.class_name);
            const [x1, y1, x2, y2] = detection.bbox_xyxy;
            const left = x1 * scaleX;
            const top = y1 * scaleY;
            const boxWidth = (x2 - x1) * scaleX;
            const boxHeight = (y2 - y1) * scaleY;
            const isActive = hasActive && index === state.activeIndex;
            const dim = hasActive && !isActive;
            const canvasName = getOfficialName(detection.class_name);
            const label = `${index + 1}. ${canvasName}`;

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
        revealHandle.classList.toggle("is-visible", hasImage);
        if (!hasImage) {
            return;
        }
        const rect = previewImage.getBoundingClientRect();
        const mediaRect = viewerMedia.getBoundingClientRect();
        const offsetLeft = rect.left - mediaRect.left;
        const x = offsetLeft + rect.width * state.revealRatio;
        revealHandle.style.left = `${x}px`;
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
                '<tr class="cost-placeholder"><td colspan="4">분석 결과가 생기면 병변별 진료항목, 비용, 자기부담률이 여기에 정리됩니다.</td></tr>';
            treatmentTotal.textContent = "-";
            return;
        }

        let totalMin = 0;
        let totalMax = 0;
        let pricedCount = 0;
        const cards = [];

        const grouped = detections.reduce((acc, item) => {
            const key = getTreatmentOfficialName(item.class_name);
            acc[key] = (acc[key] || 0) + 1;
            return acc;
        }, {});
        treatmentSummary.textContent = Object.entries(grouped)
            .map(([name, count]) => `${name} ${count}건`)
            .join(" / ");

        detections.forEach((item, index) => {
            const estimate = item.treatment_estimate || null;
            const treatmentItems = estimate && Array.isArray(estimate.treatment_items)
                ? estimate.treatment_items
                : estimate && Array.isArray(estimate.items)
                    ? estimate.items
                    : [];
            const lesionKey = `lesion-${index}`;
            const lesionLabel = `#${index + 1} ${escapeHtml(getTreatmentDisplayName(item.class_name))}`;

            if (!treatmentItems.length) {
                cards.push(`
                    <tr class="lesion-card-row">
                        <td colspan="4">
                            <section class="lesion-card">
                                <div class="lesion-card__header">
                                    <strong class="lesion-card__title">${lesionLabel}</strong>
                                </div>
                                <div class="lesion-card__empty">저장된 공공 수가 정보가 없습니다. 치과 상담을 통해 확인하세요.</div>
                            </section>
                        </td>
                    </tr>
                `);
                return;
            }

            const visibleItems = [];

            treatmentItems.forEach((treatment, treatmentIndex) => {
                const allOptions = Array.isArray(treatment.options) ? treatment.options : [];
                const options = getFilteredOptions(allOptions);
                const isFiltered = options !== allOptions;
                const filteredPrices = options.map((o) => toFiniteNumber(o.price)).filter(Number.isFinite);
                const allowFallbackPrice = !isCostFilterActive();
                const itemMin = filteredPrices.length
                    ? Math.min(...filteredPrices)
                    : allowFallbackPrice
                        ? toFiniteNumber(treatment.fee_min ?? treatment.price ?? treatment.unit_price)
                        : NaN;
                const itemMax = filteredPrices.length
                    ? Math.max(...filteredPrices)
                    : allowFallbackPrice
                        ? toFiniteNumber(treatment.fee_max ?? treatment.price ?? treatment.unit_price)
                        : NaN;
                visibleItems.push({
                    treatment,
                    treatmentIndex,
                    options,
                    isFiltered,
                    itemMin,
                    itemMax,
                });
            });

            const primaryItem = visibleItems[0] || null;
            const scopedItems = state.treatmentScopeFilter === "primary_only" ? visibleItems.slice(0, 1) : visibleItems;
            const followupItems = state.treatmentScopeFilter === "primary_only" ? [] : visibleItems.slice(1);
            const followupToggleId = `${lesionKey}-followups`;
            const followupExpanded = Boolean(state.expandedLesionCards[lesionKey]);
            const routeLabel = estimate?.followup_display_label || item.followup_display_label || "";
            const routeDescription = estimate?.followup_description || item.followup_description || "";

            const renderTreatmentItem = (itemData, roleLabel, roleClass) => {
                const { treatment, treatmentIndex, options, isFiltered, itemMin, itemMax } = itemData;
                const detailId = `${lesionKey}-detail-${treatmentIndex}`;
                const hasOptions = options.length > 1;
                return `
                    <div class="lesion-treatment lesion-treatment--${roleClass}">
                        <div class="lesion-treatment__role">${escapeHtml(roleLabel)}</div>
                        <div class="lesion-treatment__body">
                            <div class="lesion-treatment__copy">
                                <strong class="lesion-treatment__name">${escapeHtml(
                                    treatment.name || treatment.kor_nm || estimate.treatment_name || "치과 전문의 상담 필요"
                                )}</strong>
                            </div>
                            <div class="lesion-treatment__meta">
                                <div class="lesion-treatment__cost">${Number.isFinite(itemMin) || Number.isFinite(itemMax) ? escapeHtml(fmtRange(itemMin, itemMax)) : "-"}</div>
                                <div class="lesion-treatment__copay">${escapeHtml(treatment.copay_rate || "미산정")}</div>
                            </div>
                        </div>
                        ${hasOptions ? `
                            <button class="treatment-toggle" type="button" data-target="${detailId}" aria-expanded="false" aria-label="세부 옵션 보기">
                                <span class="toggle-label">${options.length}개 옵션${isFiltered ? " (필터)" : ""}</span>
                                <svg class="toggle-icon" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M2 4L6 8L10 4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
                            </button>
                            <div id="${detailId}" class="treatment-options-panel">
                                <div class="treatment-options-wrapper">
                                    <div class="treatment-options-inner">
                                        <div class="treatment-options">
                                            ${options.map((option) => {
                                                const optionPrice = toFiniteNumber(option.price);
                                                const optionMeta = buildOptionMeta(option);
                                                return `
                                                    <div class="treatment-option">
                                                        <div class="treatment-option__copy">
                                                            <span>${escapeHtml(option.name || option.full_name || "세부 옵션")}</span>
                                                            ${optionMeta ? `<small>${escapeHtml(optionMeta)}</small>` : ""}
                                                        </div>
                                                        <strong>${Number.isFinite(optionPrice) ? escapeHtml(fmtCurrency(optionPrice)) : "-"}</strong>
                                                    </div>
                                                `;
                                            }).join("")}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ` : ""}
                    </div>
                `;
            };

            const scopedMins = scopedItems.map((entry) => entry.itemMin).filter(Number.isFinite);
            const scopedMaxes = scopedItems.map((entry) => entry.itemMax).filter(Number.isFinite);
            const detMin = scopedMins.length ? Math.min(...scopedMins) : NaN;
            const detMax = scopedMaxes.length ? Math.max(...scopedMaxes) : NaN;
            if (Number.isFinite(detMin) || Number.isFinite(detMax)) {
                pricedCount += 1;
            }
            if (Number.isFinite(detMin)) totalMin += detMin;
            if (Number.isFinite(detMax)) totalMax += detMax;

            cards.push(`
                <tr class="lesion-card-row">
                    <td colspan="4">
                        <section class="lesion-card">
                            <div class="lesion-card__header">
                                <div>
                                    <strong class="lesion-card__title">${lesionLabel}</strong>
                                    <div class="lesion-card__summary">${Number.isFinite(detMin) || Number.isFinite(detMax) ? escapeHtml(fmtRange(detMin, detMax)) : "-"}</div>
                                    ${routeLabel ? `
                                        <div class="lesion-card__route">
                                            <span class="lesion-route-chip">${escapeHtml(routeLabel)}</span>
                                            ${routeDescription ? `<div class="lesion-card__route-copy">${escapeHtml(routeDescription)}</div>` : ""}
                                        </div>
                                    ` : ""}
                                </div>
                                ${followupItems.length ? `
                                    <button class="lesion-followup-toggle${followupExpanded ? " is-open" : ""}" type="button" data-target="${followupToggleId}" aria-expanded="${followupExpanded ? "true" : "false"}">
                                        <span>${followupExpanded ? "추가 진료항목 접기" : `추가 진료항목 ${followupItems.length}개 보기`}</span>
                                        <svg class="toggle-icon" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M2 4L6 8L10 4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
                                    </button>
                                ` : ""}
                            </div>
                            <div class="lesion-card__items">
                                ${primaryItem ? renderTreatmentItem(primaryItem, "대표 진료항목", "primary") : ""}
                                ${followupItems.length ? `
                                    <div id="${followupToggleId}" class="lesion-followups${followupExpanded ? " is-open" : ""}">
                                        ${followupItems.map((itemData) => renderTreatmentItem(itemData, "추가 진료항목", "followup")).join("")}
                                    </div>
                                ` : ""}
                            </div>
                        </section>
                    </td>
                </tr>
            `);
        });

        treatmentTableBody.innerHTML = cards.join("");

        treatmentTotal.textContent = pricedCount ? fmtRange(totalMin, totalMax) : "-";
    };

    const buildSeverityChip = (detection) => {
        const lesionClassName = detection.class_name;
        const severityClassName = detection.severity_class_name;
        if (!severityClassName) {
            return "";
        }
        let label = "";
        let chipClass = "is-uncertain";

        if (PERIODONTAL_CLASS_NAMES.has(lesionClassName)) {
            const stageLabel = PERIODONTAL_STAGE_LABELS[severityClassName];
            if (!stageLabel) {
                return `<span class="severity-chip ${chipClass}">치주 중증도 분류 보류</span>`;
            }
            const lesionLabel = lesionClassName === "bone_loss" ? "치조골 소실" : "치근 이개부 병변";
            label = `${stageLabel} ${lesionLabel}`;
            chipClass = severityClassName === "severe" ? "is-severe" : severityClassName === "medium" ? "is-medium" : "is-mild";
            return `<span class="severity-chip ${chipClass}">${escapeHtml(label)}</span>`;
        }

        return "";
    };

    const buildFollowupChip = (detection) => {
        if (detection.class_name !== "periapical_lesion") {
            return "";
        }
        const label = detection.followup_display_label;
        if (!label) {
            return "";
        }
        const chipClass = detection.followup_source === "model" ? "is-route" : "is-route-default";
        return `<span class="severity-chip ${chipClass}">${escapeHtml(label)}</span>`;
    };

    const buildClassExplanationCard = (className, detectionCountForClass) => {
        const info = getDisplayInfo(className);
        if (!info || !info.patient_explanation) {
            return "";
        }

        const officialName = getOfficialName(className);
        const nextStepCopy = info.patient_next_step;
        const countMarkup = detectionCountForClass
            ? `<span class="class-explanation__count">\uD0D0\uC9C0 ${escapeHtml(String(detectionCountForClass))}\uAC74</span>`
            : "";

        return `
            <section class="class-explanation">
                <div class="class-explanation__header">
                    <div class="class-explanation__eyebrow">\uBCD1\uBCC0\uBA85</div>
                    <div class="class-explanation__title-row">
                        <strong class="class-explanation__title">${escapeHtml(officialName)}</strong>
                        ${countMarkup}
                    </div>
                </div>
                <p class="class-explanation__body">${escapeHtml(info.patient_explanation)}</p>
                ${nextStepCopy ? `<p class="patient-next-step">\uB2E4\uC74C \uB2E8\uACC4: ${escapeHtml(nextStepCopy)}</p>` : ""}
            </section>
        `;
    };

    const renderList = () => {
        if (!state.detections.length) {
            summaryStrip.textContent = "아직 분석 결과가 없습니다.";
            detectionList.innerHTML =
                '<div class="placeholder-row">탐지 결과가 생기면 병변명과 위치 설명이 여기에 정리됩니다.</div>';
            detectionList.classList.remove("has-active");
            detectionCount.textContent = "--";
            state.sortedDetections = [];
            state.activeIndex = null;
            renderTreatmentTable([]);
            updateRevealHandle();
            drawDetections();
            return;
        }
        showResultSections();

        const sorted = [...state.detections].sort((a, b) => b.confidence - a.confidence);
        state.sortedDetections = sorted;
        if (state.activeIndex != null && state.activeIndex >= sorted.length) {
            state.activeIndex = null;
        }
        const grouped = sorted.reduce((acc, item) => {
            const key = getOfficialName(item.class_name);
            acc[key] = (acc[key] || 0) + 1;
            return acc;
        }, {});

        detectionCount.textContent = String(sorted.length);
        summaryStrip.textContent = Object.entries(grouped)
            .map(([name, count]) => `${name} ${count}건`)
            .join(" / ");

        // 클래스별로 그룹화 (첫 등장 순서 유지)
        const classGroups = new Map();
        sorted.forEach((item, index) => {
            const key = item.class_name;
            if (!classGroups.has(key)) classGroups.set(key, []);
            classGroups.get(key).push({ item, index });
        });

        const explanationCards = [];
        const detectionRows = [];
        for (const [className, entries] of classGroups.entries()) {
            explanationCards.push(buildClassExplanationCard(className, entries.length));
            for (const { item, index } of entries) {
                const severityChip = buildSeverityChip(item);
                const followupChip = buildFollowupChip(item);
                const officialName = getOfficialName(item.class_name);
                const badgesMarkup = [severityChip, followupChip].filter(Boolean).join("");
                const badges = badgesMarkup
                    ? `<div class="detection-badges">${badgesMarkup}</div>`
                    : "";
                detectionRows.push(`
                    <div class="detection-row" data-index="${index}" role="button" tabindex="0">
                        <div class="detection-main">
                            <span class="result-dot" style="background:${getClassColor(item.class_name)};"></span>
                            <div>
                                <strong>${index + 1}. ${escapeHtml(officialName)}</strong>
                                <div class="detection-meta">부위: ${escapeHtml(getAreaLabel(item.bbox_xyxy))}</div>
                                ${badges}
                            </div>
                        </div>
                    </div>
                `);
            }
        }
        detectionList.innerHTML = `${explanationCards.join("")}${detectionRows.join("")}`;

        updateActiveHighlight();
        renderTreatmentTable(sorted);
        updateRevealHandle();
        drawDetections();
    };

    const resetRevealToFull = () => {
        state.revealRatio = 1;
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
        state.expandedLesionCards = {};
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
        state.expandedLesionCards = {};
        resetRevealToFull();
        fileInput.value = "";
        previewImage.removeAttribute("src");
        fileName.textContent = "아직 선택하지 않음";
        fileSize.textContent = "-";
        updateViewerVisibility();
        renderList();
        hideResultSections();
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
            setStatus("탐지된 부위의 후속 진료 정보와 치주 단계를 정리 중입니다.", "busy");
            setStages("refine");
            const contentType = response.headers.get("content-type") || "";
            const rawBody = await response.text();
            const payload = contentType.includes("application/json")
                ? JSON.parse(rawBody)
                : { error: rawBody.slice(0, 240) || "Non-JSON response returned" };

            if (!response.ok) {
                throw new Error(payload.error || "Prediction request failed");
            }

            state.detections = Array.isArray(payload.detections) ? payload.detections : [];
            state.activeIndex = null;
            state.expandedLesionCards = {};
            resetRevealToFull();
            setStages("render");
            renderList();

            if (state.detections.length) {
                setStatus(
                    `분석이 완료되었습니다. ${state.detections.length}개의 의심 부위를 시각화했습니다.`,
                );
            } else {
                setStatus(
                    "분석이 완료되었습니다. 현재 기준에서는 탐지된 의심 부위가 없습니다.",
                );
            }
            window.setTimeout(() => setStages(null), 1200);
        } catch (error) {
            state.detections = [];
            state.expandedLesionCards = {};
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

    if (revealHandle) {
        let revealDragging = false;

        revealHandle.addEventListener("pointerdown", (event) => {
            revealDragging = true;
            revealHandle.setPointerCapture(event.pointerId);
            event.preventDefault();
        });

        revealHandle.addEventListener("pointermove", (event) => {
            if (!revealDragging) {
                return;
            }
            const rect = previewImage.getBoundingClientRect();
            const x = event.clientX - rect.left;
            state.revealRatio = Math.max(0, Math.min(1, x / rect.width));
            drawDetections();
            updateRevealHandle();
        });

        revealHandle.addEventListener("pointerup", () => {
            revealDragging = false;
        });

        revealHandle.addEventListener("pointercancel", () => {
            revealDragging = false;
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

    treatmentTableBody.addEventListener("click", (event) => {
        const followupButton = event.target.closest(".lesion-followup-toggle");
        if (followupButton) {
            const targetId = followupButton.dataset.target;
            const followupPanel = targetId ? document.getElementById(targetId) : null;
            if (!followupPanel) {
                return;
            }
            const expand = !followupPanel.classList.contains("is-open");
            followupPanel.classList.toggle("is-open", expand);
            followupButton.classList.toggle("is-open", expand);
            followupButton.setAttribute("aria-expanded", expand ? "true" : "false");
            const match = targetId.match(/^(lesion-\d+)-followups$/);
            if (match) {
                state.expandedLesionCards[match[1]] = expand;
            }
            const label = followupButton.querySelector("span");
            if (label) {
                const count = followupPanel.querySelectorAll(".lesion-treatment--followup").length;
                label.textContent = expand ? "추가 진료항목 접기" : `추가 진료항목 ${count}개 보기`;
            }
            return;
        }

        const button = event.target.closest(".treatment-toggle");
        if (!button) {
            return;
        }
        const targetId = button.dataset.target;
        const detailsPanel = targetId ? document.getElementById(targetId) : null;
        if (!detailsPanel) {
            return;
        }
        const expand = !detailsPanel.classList.contains("is-open");
        detailsPanel.classList.toggle("is-open", expand);
        button.classList.toggle("is-open", expand);
        button.setAttribute("aria-expanded", expand ? "true" : "false");
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

    const filterBar = document.getElementById("cost-filter-bar");
    const filterHint = document.getElementById("filter-hint");
    if (filterBar) {
        filterBar.addEventListener("click", (event) => {
            const pill = event.target.closest(".filter-pill");
            if (!pill) {
                return;
            }
            const filterType = pill.dataset.filter;
            const filterValue = pill.dataset.value;
            if (filterType === "hospital") {
                state.hospitalFilter = filterValue;
                filterBar.querySelectorAll('[data-filter="hospital"]').forEach((p) => {
                    p.classList.toggle("is-active", p.dataset.value === filterValue);
                });
            } else if (filterType === "surgery") {
                state.surgeryFilter = filterValue;
                filterBar.querySelectorAll('[data-filter="surgery"]').forEach((p) => {
                    p.classList.toggle("is-active", p.dataset.value === filterValue);
                });
            } else if (filterType === "disability") {
                state.disabilityFilter = filterValue;
                filterBar.querySelectorAll('[data-filter="disability"]').forEach((p) => {
                    p.classList.toggle("is-active", p.dataset.value === filterValue);
                });
            } else if (filterType === "treatment-scope") {
                state.treatmentScopeFilter = filterValue;
                filterBar.querySelectorAll('[data-filter="treatment-scope"]').forEach((p) => {
                    p.classList.toggle("is-active", p.dataset.value === filterValue);
                });
            }
            updateFilterHint();
            renderTreatmentTable(state.sortedDetections);
        });
    }

    updateFilterHint();

    renderLegend();
    updateViewerVisibility();
    renderList();
})();
