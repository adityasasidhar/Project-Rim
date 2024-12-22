// Frontend functionality for sending messages and interacting with the backend.
function sendMessage() {
    const userInput = document.getElementById('user-input');
    const message = userInput.value.trim();

    if (!message) {
        alert("Please enter a message!");
        return;
    }

    // Append user's message to the chat box
    appendMessage(message, "user-message");

    // Clear the input field
    userInput.value = "";

    // Send the message to the backend
    fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: message }),
    })
        .then(response => response.json())
        .then(data => {
            if (data.response) {
                // Append the AI's response to the chat box
                appendMessage(data.response, "ai-message");
            } else {
                console.error("Invalid response from server:", data);
            }
        })
        .catch(error => console.error("Error communicating with server:", error));
}

function appendMessage(message, className) {
    const chatBox = document.getElementById('chat-box');
    const messageDiv = document.createElement('div');
    messageDiv.className = className;
    messageDiv.innerText = message;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Handle 'Enter' key to send message
function checkEnter(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Set current date/time in the header
document.getElementById('date-time').innerText = new Date().toLocaleString();