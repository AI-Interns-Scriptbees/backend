/**
 * ScriptBees Assistant - Frontend
 * Clean Final Version (No Errors, Fully Compatible)
 */

// ==========================================
// CONFIGURATION
// ==========================================

const API_URL = 'http://localhost:8000'; // Backend URL
let API_KEY = 'X2Cli1ZSPhHHAHlfZkOEPRWIqtd1TQD9ErH705-HMc4'; // Your API key

// ==========================================
// DOM ELEMENTS
// ==========================================

const chatMessages = document.getElementById('chat-container');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-btn');

// ==========================================
// FIX: Add a safe speakText() function
// ==========================================

function speakText(text) {
    try {
        const utter = new SpeechSynthesisUtterance(text);
        utter.rate = 1.0;
        utter.pitch = 1.0;
        speechSynthesis.speak(utter);
    } catch (e) {
        console.warn("Speech synthesis not available:", e);
    }
}

// ==========================================
// MESSAGE FUNCTIONS
// ==========================================

function addMessage(text, isUser = false, sources = null, isError = false) {
    const msg = document.createElement('div');
    msg.className = `message ${isUser ? 'user' : 'bot'}`;

    const content = document.createElement('div');
    content.className = 'message-content';
    if (isError) content.style.color = 'red';

    const textElem = document.createElement('p');
    textElem.innerHTML = text.replace(/\n/g, '<br>');
    content.appendChild(textElem);

    // Add sources
    if (sources && sources.length > 0) {
        const srcDiv = document.createElement('div');
        srcDiv.className = 'sources';
        srcDiv.innerHTML = "<b>📌 Sources:</b><br>";

        sources.forEach(url => {
            const link = document.createElement('a');
            link.href = url;
            link.textContent = url;
            link.target = "_blank";
            srcDiv.appendChild(link);
            srcDiv.appendChild(document.createElement('br'));
        });

        content.appendChild(srcDiv);
    }

    msg.appendChild(content);
    chatMessages.appendChild(msg);

    // Auto speak for bot replies
    if (!isUser && !isError) {
        speakText(text);
    }

    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
}

function showTyping() {
    const typing = document.createElement('div');
    typing.className = 'message bot';
    typing.id = 'typing-indicator';

    const content = document.createElement('div');
    content.className = 'typing-dots';

    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.className = 'typing-dot';
        content.appendChild(dot);
    }

    typing.appendChild(content);
    chatMessages.appendChild(typing);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTyping() {
    const typing = document.getElementById('typing-indicator');
    if (typing) typing.remove();
}

// ==========================================
// API FUNCTIONS
// ==========================================

async function sendQuery(question) {
    try {
        const response = await fetch(`${API_URL}/api/ask`, {
            method: "POST",
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': API_KEY
            },
            body: JSON.stringify({ question })
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || err.error || 'Server error');
        }

        return await response.json();

    } catch (err) {
        if (err.message.includes('Failed to fetch')) {
            throw new Error(
                `Cannot connect to backend at ${API_URL}\n` +
                `Check:\n1. Backend running\n2. CORS enabled\n3. API key correct`
            );
        }
        throw err;
    }
}

async function checkHealth() {
    try {
        const res = await fetch(`${API_URL}/health`);
        const data = await res.json();

        console.log('Server health:', data);

        if (!data.retriever_loaded || !data.generator_loaded) {
            addMessage("⚠️ Loading… ScriptBees models still initializing.", false);
        }

    } catch {
        addMessage("❌ Unable to reach backend. Check server.", false, null, true);
    }
}

// ==========================================
// FORM HANDLING
// ==========================================

async function handleSend() {
    const question = userInput.value.trim();
    if (!question) return;

    addMessage(question, true);

    userInput.value = "";
    userInput.disabled = true;
    sendButton.disabled = true;

    showTyping();

    try {
        const response = await sendQuery(question);
        removeTyping();
        addMessage(response.answer, false, response.sources);

    } catch (err) {
        removeTyping();
        addMessage(`⚠️ Error:\n${err.message}`, false, null, true);

    } finally {
        userInput.disabled = false;
        sendButton.disabled = false;
    }
}

// ==========================================
// INITIALIZATION
// ==========================================

function init() {

    sendButton.addEventListener("click", handleSend);

    userInput.addEventListener("keydown", e => {
        if (e.key === "Enter") {
            e.preventDefault();
            handleSend();
        }
    });

    console.log("Frontend initialized");
    console.log(`Backend: ${API_URL}`);
    console.log(`API Key prefix: ${API_KEY.substring(0, 6)}...`);

    checkHealth();

    setTimeout(() => {
        addMessage("🐝 Hi! I'm **ScriptBees Assistant**.\nAsk me anything about ScriptBees!", false);
    }, 200);
}

document.addEventListener("DOMContentLoaded", init);

// Allow key updates externally
window.updateApiKey = function(newKey) {
    API_KEY = String(newKey || "").trim();
    console.log("API key updated:", API_KEY.substring(0, 8) + "...");
};
