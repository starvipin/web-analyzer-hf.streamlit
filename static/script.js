const urlInput = document.getElementById("urlInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const statusContainer = document.getElementById("status-container");
const statusMsg = document.getElementById("status-msg");
const spinner = document.getElementById("loading-spinner");
const landingPage = document.getElementById("landing-page");
const chatPage = document.getElementById("chat-page");
const activeUrlText = document.getElementById("activeUrlText");

// Preview Card Elements
const previewImg = document.getElementById("previewImg");
const previewEmoji = document.getElementById("previewEmoji");
const previewTitle = document.getElementById("previewTitle");
const previewDesc = document.getElementById("previewDesc");

const chatInput = document.getElementById("chatInput");
const chatBox = document.getElementById("chatBox");
const sendBtn = document.getElementById("sendBtn");
const sessionId = getSessionId();

function getSessionId() {
    const key = "linkmindSessionId";
    let existing = window.localStorage.getItem(key);
    if (existing) return existing;

    const generated = window.crypto?.randomUUID
        ? window.crypto.randomUUID()
        : "session-" + Date.now() + "-" + Math.random().toString(16).slice(2);
    window.localStorage.setItem(key, generated);
    return generated;
}

// --- URL INGESTION & METADATA LOGIC ---
async function processUrl() {
    const url = urlInput.value.trim();
    
    // CHANGE: Agar input khali hai toh soft grey message dikhega
    if (url === "") {
        showStatus("✨ Please enter a link to begin", "#a1a1aa", false);
        return;
    }
    
    let parsedUrl;
    try {
        parsedUrl = new URL(url);
    } catch (e) {
        showStatus("✨ Please enter a valid URL", "#a1a1aa", false);
        return;
    }

    if (!["http:", "https:"].includes(parsedUrl.protocol)) {
        showStatus("✨ Make sure your link starts with http:// or https://", "#a1a1aa", false);
        return;
    }
    
    // Add Glow & Loading state to Button
    analyzeBtn.classList.add("is-loading");
    analyzeBtn.disabled = true;
    
    showStatus("Initializing Neural Engine...", "#60a5fa", true);

    // Default fallback metadata
    let metaTitle = parsedUrl.hostname;
    let metaDesc = "Analyzed context from the provided link.";
    let imageUrl = null;

    try {
        statusMsg.innerText = "Fetching Metadata & Preview...";
        
        // Fetching Metadata from Microlink
        const metaRes = await fetch(`https://api.microlink.io/?url=${encodeURIComponent(url)}`);
        const metaData = await metaRes.json();
        
        if (metaData.status === 'success') {
            metaTitle = metaData.data.title || metaTitle;
            metaDesc = metaData.data.description || metaDesc;
            imageUrl = metaData.data.image?.url || metaData.data.logo?.url;
        }
    } catch (e) {
        console.log("Meta preview failed, using defaults");
    }

    try {
        statusMsg.innerText = "Extracting knowledge base...";
        
        const response = await fetch('/api/ingest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url, session_id: sessionId })
        });
        const ingestData = await response.json();

        if (!response.ok) {
            throw new Error(ingestData.detail || "Failed to analyze this URL.");
        }
        
        showStatus("System Ready! Connecting...", "#34d399", false);
        
        // Update Link Preview Card UI
        previewTitle.innerText = metaTitle;
        previewDesc.innerText = metaDesc;
        if(imageUrl) {
            previewImg.src = imageUrl;
            previewImg.style.display = "block";
            previewEmoji.style.display = "none";
        } else {
            previewImg.style.display = "none";
            previewEmoji.style.display = "block";
        }

        // Smooth Transition to Chat Page
        setTimeout(() => {
            landingPage.classList.add("page-exit");
            setTimeout(() => {
                landingPage.style.display = "none";
                chatPage.style.display = "flex";
                
                void chatPage.offsetWidth; 
                chatPage.classList.add("page-enter");
                
                activeUrlText.innerText = "Synched: " + parsedUrl.hostname;
                
                // Bot Greeting
                if(chatBox.children.length === 0) {
                    setTimeout(() => {
                        addMessage("Hello! I have analyzed the link. What would you like to know?", "bot");
                    }, 500);
                }
            }, 600);
        }, 800);
        
    } catch (error) {
        showStatus("Connection failed: " + error.message, "#ef4444", false);
        resetAnalyzeBtn();
    }
}

urlInput.addEventListener("keypress", e => { if (e.key === "Enter") processUrl(); });
analyzeBtn.addEventListener("click", processUrl);

function showStatus(text, color, showSpinner) {
    statusContainer.style.opacity = "1";
    statusMsg.innerText = text;
    statusMsg.style.color = color;
    spinner.style.display = showSpinner ? "block" : "none";
}

function resetAnalyzeBtn() {
    analyzeBtn.classList.remove("is-loading");
    analyzeBtn.disabled = false;
}

// --- CHAT LOGIC ---
async function handleChat() {
    const query = chatInput.value.trim();
    if (query === "") return;
    
    chatInput.value = "";
    addMessage(query, "user");
    
    const botId = "bot-loading-" + Date.now();
    addMessage("Thinking...", "bot", botId);

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, session_id: sessionId })
        });
        const data = await response.json();
        
        if(response.ok) {
            document.getElementById(botId).innerText = data.answer;
        }
        else document.getElementById(botId).innerText = "Error: " + (data.detail || "Failed");
    } catch (error) {
        document.getElementById(botId).innerText = "Failed to connect to server.";
    }
}

chatInput.addEventListener("keypress", e => { if (e.key === "Enter") handleChat(); });
sendBtn.addEventListener("click", handleChat);

function addMessage(text, sender, id = null) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `msg ${sender}`;
    msgDiv.innerText = text;
    if (id) msgDiv.id = id;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// --- RESET APP ---
function resetApp() {
    chatPage.classList.remove("page-enter");
    setTimeout(() => {
        chatPage.style.display = "none";
        landingPage.style.display = "flex";
        
        void landingPage.offsetWidth;
        landingPage.classList.remove("page-exit");
        
        urlInput.value = "";
        statusContainer.style.opacity = "0";
        chatBox.innerHTML = "";
        previewImg.src = "";
        previewImg.style.display = "none";
        previewEmoji.style.display = "block";
        resetAnalyzeBtn();
    }, 600);
}

window.processUrl = processUrl;
window.resetApp = resetApp;
