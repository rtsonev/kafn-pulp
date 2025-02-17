getResultMessage = (prob, label) => {
    /**
     * Returns a result message and icon
     */
    let message = "";
    let iconHTML = "";
    
    if (label === "Fake") {
      if (prob < 70) {
        message = "The news is predicted as FAKE (low confidence).";
        iconHTML = '<img src="assets/warning.png" alt="Warning" class="icon">';
      } else if (prob < 90) {
        message = "The news is predicted as FAKE (most likely fake).";
        iconHTML = '<img src="assets/stop.png" alt="Stop" class="icon">';
      } else {
        message = "The news is predicted as FAKE (extremely fake).";
        iconHTML = '<img src="assets/cross.png" alt="Cross" class="icon">';
      }
    } else {
      if (prob < 30) {
        message = "The news is predicted as REAL (high confidence).";
        iconHTML = '<img src="assets/check.png" alt="Check" class="icon">';
      } else {
        message = "The news is predicted as REAL (but with some uncertainty).";
        iconHTML = '<img src="assets/thinking.png" alt="Thinking" class="icon">';
      }
    }
    return { message, iconHTML };
  }
  
getPrediction = async (text) => {
    /**
     * Get prediction of a text from API
     */
    const response = await fetch("http://192.168.1.4:5555/predict", { 
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    });
    if (!response.ok) {
      throw new Error(`ERROR! status: ${response.status}`);
    }
    const data = await response.json();
    return data;
  }
  
updateResult = (content) => {
    /**
     * Update the result element in the popup window
     */
    document.getElementById("result").innerHTML = content;
  }

handleClick = async () => {
    /**
     * Main event listener for the classify button
     */
    const newsText = document.getElementById("newsText").value.trim();
    if (!newsText) {
      updateResult("Please enter some text.");
      return;
    }
    updateResult("Classifying...");
    
    try {
      const data = await getPrediction(newsText);
      const label = data["test result"]["label"] === 1 ? "Fake" : "Real";
      const prob = (data["test result"]["probability"] * 100).toFixed(2);
      const { message, iconHTML } = getResultMessage(prob, label);
      updateResult(`
        <div>${iconHTML} ${message}</div>
        <div><strong>Label:</strong> ${label}</div>
        <div><strong>Probability to be fake:</strong> ${prob}%</div>
      `);
    } catch (err) {
      console.error(err);
      updateResult("Error classifying news.");
    }
  }

document.getElementById("classifyButton").addEventListener("click", handleClick);
  