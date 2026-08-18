const input = document.getElementById("input");
const chatbox = document.getElementById("chatbox");

function appendMessage(sender, text) {//show messages on screen
  const message = document.createElement("div");
  message.classList.add("message");
  message.classList.add(sender === "You" ? "user" : "bot");
  message.textContent = text;
  chatbox.appendChild(message);
  chatbox.scrollTop = chatbox.scrollHeight;
}

async function sendMessage() {//send and fetch reply
  const message = input.value.trim();
  if (!message) return;

  appendMessage("You", message);
  input.value = "";

  try {
    const res = await fetch("http://127.0.0.1:8000/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message })
    });

    const data = await res.json();
    const botReply = data.response || data.answer || "Sorry, no response.";
    appendMessage("Bot", botReply);
  } catch (error) {
    appendMessage("Bot", "Error contacting server.");
  }
}
