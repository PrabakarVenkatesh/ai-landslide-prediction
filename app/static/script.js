let selectedFile = null;

const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");

dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.style.background = "#1e293b";
});

dropArea.addEventListener("dragleave", () => {
    dropArea.style.background = "transparent";
});

dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    selectedFile = e.dataTransfer.files[0];
    showPreview(selectedFile);
});

fileInput.addEventListener("change", () => {
    selectedFile = fileInput.files[0];
    showPreview(selectedFile);
});

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = () => {
        preview.src = reader.result;
        preview.style.display = "block";
    };
    reader.readAsDataURL(file);
}

async function uploadImage() {
    if (!selectedFile) {
        alert("Please select an image");
        return;
    }

    document.getElementById("loading").style.display = "block";

    let formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model", document.getElementById("modelSelect").value);

    let response = await fetch("/", {
        method: "POST",
        body: formData
    });

    let text = await response.text();

    document.getElementById("loading").style.display = "none";
    document.getElementById("result").innerHTML = text;
}