// Function to send a message
function sendMessage() {
    const userInput = document.getElementById('user-input');
    const message = userInput.value.trim();

    if (!message) {
        alert("Please enter a message!");
        return;
    }

    // Append user's message
    appendMessage(message, "floating-message user-message");

    // Clear the input field
    userInput.value = "";

    // Simulate sending the message to a backend
    fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: message }),
    })
        .then(response => response.json())
        .then(data => {
            if (data.response) {
                // Append the AI's response
                appendMessage(data.response, "floating-message ai-message");
            } else {
                console.error("Invalid response from server:", data);
            }
        })
        .catch(error => console.error("Error communicating with server:", error));
}

// Function to append a message
function appendMessage(message, className) {
    const chatBackground = document.getElementById('chat-background');
    const messageDiv = document.createElement('div');
    messageDiv.className = className;
    messageDiv.innerText = message;

    chatBackground.appendChild(messageDiv);

    // Automatically scroll to the latest message
    chatBackground.scrollTop = chatBackground.scrollHeight;
}

// Live Date and Time
function updateDateTime() {
    const now = new Date();
    const options = { day: 'numeric', month: 'long', year: 'numeric' };
    const date = now.toLocaleDateString('en-US', options);
    const time = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    document.getElementById('date-time').innerText = `${date}, ${time}`;
}

// Update date and time every second
setInterval(updateDateTime, 1000);
updateDateTime();

// Handle 'Enter' key to send message
function checkEnter(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}
