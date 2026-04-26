(function () {
    "use strict";

    var API_URL = "/chat";

    var chatMessages = document.getElementById("chatMessages");
    var chatForm = document.getElementById("chatForm");
    var messageInput = document.getElementById("messageInput");
    var sendBtn = document.getElementById("sendBtn");
    var welcomeCard = document.getElementById("welcomeCard");
    var heroSection = document.getElementById("heroSection");
    var statusText = document.getElementById("statusText");
    var toast = document.getElementById("toast");

    var isSending = false;
    var typingNode = null;
    var toastTimer = null;
    var scrollFrame = null;
    var heroHidden = false;
    var chatHistory = [];

    // ---- Two-step flow state ----
    // When a chip requires follow-up input (e.g. faculty name),
    // we store the pending action here.
    var pendingAction = null;  // e.g. { type: "faculty" }

    document.addEventListener("DOMContentLoaded", init, { once: true });

    function init() {
        chatForm.addEventListener("submit", onSubmit);
        bindClick("clearBtn", clearChat);

        // Bind all sidebar chip buttons
        document.querySelectorAll(".chip").forEach(function (button) {
            button.addEventListener("click", function () {
                var action = button.getAttribute("data-action");
                var query = button.getAttribute("data-query");

                if (action) {
                    // Two-step flow (e.g. faculty search)
                    handleAction(action);
                } else if (query) {
                    // One-step flow — send the query directly
                    sendMessage(query);
                }
            });
        });

        messageInput.focus();
        scrollToBottom(false);
    }

    // ---- Two-step action handler ----
    function handleAction(action) {
        if (isSending) return;

        hideWelcome();
        hideHeroOnce();

        if (action === "faculty") {
            appendMessage("bot", "Please enter the faculty name you want to search for:");
            pendingAction = { type: "faculty" };
            messageInput.placeholder = "Type faculty name...";
            messageInput.classList.add("awaiting-input");
            messageInput.focus();
        }
    }

    function bindClick(id, handler) {
        var element = document.getElementById(id);
        if (element) {
            element.addEventListener("click", handler);
        }
    }

    function onSubmit(event) {
        event.preventDefault();
        sendMessage();
    }

    async function sendMessage(forcedText) {
        if (isSending) return;

        var text = (forcedText || messageInput.value).trim();
        if (!text) return;

        // Check if we have a pending action (two-step flow)
        if (!forcedText && pendingAction) {
            if (pendingAction.type === "faculty") {
                // Prefix with "faculty" so the backend detects the intent
                text = "faculty " + text;
            }
            // Clear pending state
            pendingAction = null;
            messageInput.placeholder = "Ask about faculty, fees, PYQs, hostels, portals, placements...";
            messageInput.classList.remove("awaiting-input");
        }

        setSending(true);
        hideHeroOnce();
        hideWelcome();
        appendMessage("user", text);
        messageInput.value = "";
        showTyping();

        try {
            var response = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text, history: chatHistory })
            });

            if (!response.ok) {
                throw new Error("Server returned " + response.status);
            }

            var data = await response.json();
            hideTyping();
            var reply = data && data.reply ? data.reply : "I could not find an answer.";
            appendMessage("bot", reply);
            
            // Push to chat history
            chatHistory.push({ role: "user", parts: [{ text: text }] });
            chatHistory.push({ role: "model", parts: [{ text: reply }] });
            
            // Keep only the last 10 turns (20 messages) to prevent huge payloads
            if (chatHistory.length > 20) {
                chatHistory = chatHistory.slice(chatHistory.length - 20);
            }
            
            setStatus("Answer returned", "ready");
        } catch (error) {
            console.error("Chat request failed:", error);
            hideTyping();
            appendMessage("bot", "I could not reach the server. Please make sure Flask is running at http://localhost:5000.");
            setStatus("Backend connection failed", "offline");
            showToast("Unable to send message. Check the backend server.");
        } finally {
            setSending(false);
            messageInput.focus();
            scrollToBottom(true);
        }
    }

    function appendMessage(role, text) {
        var row = document.createElement("div");
        row.className = "message-row " + role;

        var avatar = document.createElement("div");
        avatar.className = "message-avatar";
        avatar.textContent = role === "bot" ? "M" : "Y";

        var wrap = document.createElement("div");
        wrap.className = "bubble-wrap";

        var bubble = document.createElement("div");
        bubble.className = "message-bubble";
        bubble.innerHTML = formatText(text);

        var time = document.createElement("div");
        time.className = "message-time";
        time.textContent = getTime();

        wrap.appendChild(bubble);
        wrap.appendChild(time);
        row.appendChild(avatar);
        row.appendChild(wrap);
        chatMessages.appendChild(row);
        scrollToBottom(true);
    }

    function showTyping() {
        hideTyping();

        typingNode = document.createElement("div");
        typingNode.className = "typing-row";

        var avatar = document.createElement("div");
        avatar.className = "message-avatar";
        avatar.textContent = "M";

        var dots = document.createElement("div");
        dots.className = "typing-dots";
        dots.setAttribute("aria-label", "Assistant is typing");
        dots.innerHTML = "<span></span><span></span><span></span>";

        typingNode.appendChild(avatar);
        typingNode.appendChild(dots);
        chatMessages.appendChild(typingNode);
        scrollToBottom(true);
    }

    function hideTyping() {
        if (typingNode && typingNode.parentNode) {
            typingNode.parentNode.removeChild(typingNode);
        }
        typingNode = null;
    }

    function scrollToBottom(smooth) {
        if (scrollFrame) {
            cancelAnimationFrame(scrollFrame);
        }

        scrollFrame = requestAnimationFrame(function () {
            chatMessages.scrollTo({
                top: chatMessages.scrollHeight,
                behavior: smooth ? "smooth" : "auto"
            });

            requestAnimationFrame(function () {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            });
            scrollFrame = null;
        });
    }

    function clearChat() {
        hideTyping();
        chatMessages.querySelectorAll(".message-row, .typing-row").forEach(function (node) {
            node.remove();
        });

        if (welcomeCard) {
            welcomeCard.style.display = "";
        }

        // Reset pending action
        pendingAction = null;
        messageInput.placeholder = "Ask about faculty, fees, PYQs, hostels, portals, placements...";
        messageInput.classList.remove("awaiting-input");

        chatHistory = []; // Reset context
        messageInput.value = "";
        setStatus("Chat surface cleared", "ready");
        messageInput.focus();
        scrollToBottom(false);
    }

    function hideHeroOnce() {
        if (!heroHidden && heroSection) {
            heroSection.style.display = "none";
            heroHidden = true;
        }
    }

    function hideWelcome() {
        if (welcomeCard) {
            welcomeCard.style.display = "none";
        }
    }

    function setSending(value) {
        isSending = value;
        sendBtn.disabled = value;
        messageInput.disabled = value;
        sendBtn.textContent = value ? "Sending" : "Send";
        setStatus(value ? "Querying PostgreSQL search layer..." : "Connected to the Flask relay", value ? "searching" : "ready");
    }

    function setStatus(text, state) {
        if (statusText) {
            statusText.textContent = text;
        }
    }

    function formatText(text) {
        return String(text || "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(
                /(https?:\/\/[^\s<]+)/g,
                '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
            );
    }

    function getTime() {
        return new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit"
        });
    }

    function showToast(message) {
        clearTimeout(toastTimer);
        toast.textContent = message;
        toast.classList.add("show");
        toastTimer = setTimeout(function () {
            toast.classList.remove("show");
        }, 3200);
    }
})();
