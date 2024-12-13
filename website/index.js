document.getElementById("send-image").addEventListener("click", async () => {
  const imageInput = document.getElementById("image-input");
  if (imageInput.files.length > 0) {
    const file = imageInput.files[0];
    const reader = new FileReader();
    reader.onloadend = async () => {
      const base64Image = reader.result.split(",")[1]; // Get base64 part
      const response = await sendImageToServer(base64Image);
      if (response) {
        displayResponse(response);
      }
    };
    reader.readAsDataURL(file);
  } else {
    console.log("No image selected");
  }
});

document.getElementById('image-input').addEventListener('change', function (event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const img = document.getElementById('image-preview');
            img.src = e.target.result;
            img.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

async function sendImageToServer(base64Image) {
  try {
    const response = await fetch("http://localhost:9090", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image: base64Image }),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error sending image to server:", error);
  }
}

function displayResponse(response) {
    if (response === "the year is too old to get accurate material data") {
        document.getElementById("infoBuilding").innerHTML = "The year is too old to get accurate material data.";
        document.getElementById("infoMaterial").innerHTML = "No data available.";
        return;
    }
    const [area, year, materials] = response;

    let infoBuilding = `Estimated to be built in ${year} and has a Estimated surface area of ${area} m²`;
    let infoMaterial = "<h3>Materials:</h3><table><thead><tr><th>Material</th><th>Amount (kg)</th></tr></thead><tbody>";
    for (let material in materials) {
        let amount = (materials[material] * area).toFixed(2);
        amount = parseFloat(amount).toLocaleString('de-DE');
        infoMaterial += `<tr><td><strong>${material}</strong></td><td>${amount}</td></tr>`;
    }
    infoMaterial += "</tbody></table>";

    document.getElementById("infoBuilding").innerHTML = infoBuilding;
    document.getElementById("infoMaterial").innerHTML = infoMaterial;
}