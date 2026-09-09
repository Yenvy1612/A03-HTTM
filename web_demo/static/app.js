const binaryFields = [
  ["HighBP", "Cao huyết áp"], ["HighChol", "Cholesterol cao"],
  ["CholCheck", "Đã kiểm tra cholesterol"], ["Smoker", "Có hút thuốc"],
  ["Stroke", "Tiền sử đột quỵ"], ["HeartDiseaseorAttack", "Bệnh tim"],
  ["PhysActivity", "Có vận động"], ["Fruits", "Ăn trái cây"],
  ["Veggies", "Ăn rau"], ["HvyAlcoholConsump", "Uống rượu nhiều"],
  ["AnyHealthcare", "Có bảo hiểm y tế"], ["NoDocbcCost", "Bỏ khám vì chi phí"],
  ["DiffWalk", "Khó đi lại"], ["Sex", "Giới tính nam"]
];

const toggleRoot = document.querySelector("#diabetes-toggles");
binaryFields.forEach(([name, label], index) => {
  const wrapper = document.createElement("label");
  wrapper.className = "toggle";
  const checked = ["CholCheck", "PhysActivity", "Fruits", "Veggies", "AnyHealthcare"].includes(name);
  wrapper.innerHTML = `<span>${label}</span><input type="checkbox" name="${name}" ${checked ? "checked" : ""}><i aria-hidden="true"></i>`;
  toggleRoot.appendChild(wrapper);
});

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => {
      const active = item === tab;
      item.classList.toggle("active", active);
      item.setAttribute("aria-selected", String(active));
    });
    document.querySelectorAll(".panel").forEach((panel) => {
      const active = panel.id === tab.dataset.target;
      panel.classList.toggle("active", active);
      panel.hidden = !active;
    });
  });
});

const requestedTab = new URLSearchParams(window.location.search).get("tab");
if (requestedTab) {
  const requestedButton = document.querySelector(`.tab[data-target="${requestedTab}-panel"]`);
  if (requestedButton) requestedButton.click();
}

function formObject(form) {
  return Object.fromEntries(new FormData(form).entries());
}

function setLoading(form, loading) {
  const button = form.querySelector("button[type=submit]");
  button.disabled = loading;
  if (!button.dataset.label) button.dataset.label = button.innerHTML;
  button.innerHTML = loading ? "Đang chạy model…" : button.dataset.label;
}

async function postPrediction(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const data = await response.json();
  if (!response.ok || !data.ok) throw new Error(data.error || "Không thể chạy model.");
  return data;
}

function showError(root, error) {
  root.hidden = false;
  root.className = "result error";
  root.innerHTML = `<div class="result-grid"><div class="result-icon">!</div><h3>Không thể dự đoán</h3><p>${error.message}</p></div>`;
}

function percent(value) { return `${(value * 100).toFixed(1)}%`; }
function rupees(value) { return new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(value); }

const diabetesForm = document.querySelector("#diabetes-form");
diabetesForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const result = document.querySelector("#diabetes-result");
  const payload = formObject(diabetesForm);
  binaryFields.forEach(([name]) => payload[name] = diabetesForm.elements[name].checked ? 1 : 0);
  setLoading(diabetesForm, true);
  try {
    const data = await postPrediction("/api/predict/diabetes", payload);
    result.hidden = false; result.className = "result";
    result.innerHTML = `<div class="result-grid"><div class="result-icon">${data.prediction ? "!" : "✓"}</div><h3>${data.label}</h3><p>Xác suất lớp tiểu đường: ${percent(data.probability)} · ${data.model}</p></div><div class="meter-list"><div class="meter-row"><span>Nguy cơ</span><div class="meter"><i style="width:${percent(data.probability)}"></i></div><strong>${percent(data.probability)}</strong></div></div><p>${data.note}</p>`;
  } catch (error) { showError(result, error); }
  finally { setLoading(diabetesForm, false); }
});

const houseForm = document.querySelector("#house-form");
houseForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const result = document.querySelector("#house-result");
  setLoading(houseForm, true);
  try {
    const data = await postPrediction("/api/predict/house", formObject(houseForm));
    result.hidden = false; result.className = "result";
    result.innerHTML = `<div class="result-grid"><div class="result-icon">₹</div><h3>${rupees(data.price_per_sqft)} / sqft</h3><p>Tổng giá tham khảo: ${rupees(data.estimated_total)} · ${data.model}</p></div><p>${data.unit_note}</p>`;
  } catch (error) { showError(result, error); }
  finally { setLoading(houseForm, false); }
});

const sentimentForm = document.querySelector("#sentiment-form");
document.querySelectorAll("[data-review]").forEach(button => button.addEventListener("click", () => {
  sentimentForm.elements.review.value = button.dataset.review;
}));
sentimentForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const result = document.querySelector("#sentiment-result");
  setLoading(sentimentForm, true);
  try {
    const data = await postPrediction("/api/predict/sentiment", formObject(sentimentForm));
    const meters = ["Negative", "Neutral", "Positive"].map(label => `<div class="meter-row"><span>${label}</span><div class="meter"><i style="width:${percent(data.scores[label])}"></i></div><strong>${percent(data.scores[label])}</strong></div>`).join("");
    result.hidden = false; result.className = "result";
    result.innerHTML = `<div class="result-grid"><div class="result-icon">Aa</div><h3>${data.label}</h3><p>${data.model}</p></div><div class="meter-list">${meters}</div>`;
  } catch (error) { showError(result, error); }
  finally { setLoading(sentimentForm, false); }
});
