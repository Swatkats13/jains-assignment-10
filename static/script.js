document.getElementById("search-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);

    const response = await fetch("/", {
        method: "POST",
        body: formData,
    });

    const results = await response.json();
    const resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = "";

    results.forEach((result) => {
        const resultContainer = document.createElement("div");
        resultContainer.style.margin = "10px";
        resultContainer.style.textAlign = "center";

        const img = document.createElement("img");
        img.src = result.file_name;
        img.alt = "Search result";
        img.style.width = "150px";
        img.style.height = "150px";
        img.style.objectFit = "cover";

        const p = document.createElement("p");
        p.textContent = `Similarity: ${result.similarity.toFixed(3)}`;

        resultContainer.appendChild(img);
        resultContainer.appendChild(p);
        resultsDiv.appendChild(resultContainer);
    });
});
