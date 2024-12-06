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
  document.getElementById("area").innerText = `Area: ${response[0]}`;
  document.getElementById("year").innerText = `Year: ${response[1]}`;
  document.getElementById("material").innerText = `Material: ${response[2]}`;
  document.getElementById("amount").innerText = `Amount: ${response[3]}`;
    document.getElementById("stuff").innerText = `Stuff: ${response[3] * response[0]}`;

}