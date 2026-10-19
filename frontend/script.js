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
    var sessionId = makeSessionId();

    // ---- Two-step flow state ----
    // When a chip requires follow-up input (e.g. faculty name),
    // we store the pending action here.
    var pendingAction = null;  // e.g. { type: "faculty" }

    function makeSessionId() {
        if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
        return "s" + Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
    }

    var ROUTE_LABELS = {
        rule: "Quick reply", skill: "SRM database", tool: "Exact tool", "kb+llm": "SRM knowledge",
        kb: "SRM knowledge", decline: "Out of scope", safety: "Safety check"
    };

    async function checkHealth() {
        var dot = document.getElementById("dbDot");
        var readout = document.getElementById("dbReadout");
        var pill = document.getElementById("liveText");
        try {
            var res = await fetch("/health");
            var h = await res.json();
            var label = h.db_ok ? "DB LIVE" : "KB MODE";
            readout.textContent = label;
            pill.textContent = "Online";
            if (dot) dot.style.opacity = h.db_ok ? "1" : "0.45";
            setStatus(h.db_ok ? "Connected" : "Running on the built-in SRM knowledge base", "ready");
        } catch (e) {
            readout.textContent = "OFFLINE";
            pill.textContent = "Offline";
            if (dot) dot.style.opacity = "0.2";
            setStatus("Waking up the server...", "offline");
        }
    }

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

        checkHealth();
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

        // Pending two-step action: send it as an explicit action field
        var action = null;
        if (!forcedText && pendingAction) {
            action = pendingAction.type;
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
                body: JSON.stringify({ message: text, history: chatHistory, session_id: sessionId, action: action })
            });

            if (response.status === 403) {
                // blocked by the hosting firewall (e.g. raw SQL/script text) before it reaches the app
                hideTyping();
                appendMessage("bot", "That message looks like code or a technical command, so it was blocked by the security filter. Please ask a normal question about SRM - admissions, hostels, fees, PYQs, placements, library or HODs.");
                setStatus("Message blocked", "ready");
                return;
            }
            if (response.status === 400 || response.status === 429) {
                var info = await response.json().catch(function () { return {}; });
                hideTyping();
                appendMessage("bot", (info && info.reply) || "Please check your message and try again.");
                setStatus(response.status === 429 ? "Slow down a little" : "Check your message", "ready");
                return;
            }
            if (!response.ok) {
                throw new Error("Server returned " + response.status);
            }

            var data = await response.json();
            hideTyping();
            var reply = data && data.reply ? data.reply : "I could not find an answer.";
            appendMessage("bot", reply, data);

            // Push to chat history
            chatHistory.push({ role: "user", text: text });
            chatHistory.push({ role: "model", text: reply });
            
            // Keep only the last 10 turns (20 messages) to prevent huge payloads
            if (chatHistory.length > 20) {
                chatHistory = chatHistory.slice(chatHistory.length - 20);
            }
            
            setStatus("Answer returned", "ready");
        } catch (error) {
            console.error("Chat request failed:", error);
            hideTyping();
            appendMessage("bot", "I could not reach the server right now. It may be waking up - please wait a few seconds and try again.");
            setStatus("Server unreachable", "offline");
            showToast("Could not reach MIST AI. Try again in a moment.");
            checkHealth();
        } finally {
            setSending(false);
            messageInput.focus();
            scrollToBottom(true);
        }
    }

    function appendMessage(role, text, meta) {
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
        if (role === "bot" && meta && meta.route) {
            wrap.appendChild(buildMeta(meta));
        }
        wrap.appendChild(time);
        row.appendChild(avatar);
        row.appendChild(wrap);
        chatMessages.appendChild(row);
        scrollToBottom(true);
    }

    function buildMeta(meta) {
        var bar = document.createElement("div");
        bar.className = "message-meta";

        var badge = document.createElement("span");
        badge.className = "route-badge route-" + String(meta.route).replace("+", "-");
        var label = ROUTE_LABELS[meta.route] || meta.route;
        badge.textContent = label;
        badge.title = "Topic: " + (meta.intent || "n/a");
        bar.appendChild(badge);

        if (meta.chat_id) {
            [["up", "\uD83D\uDC4D", 1, "Helpful"], ["down", "\uD83D\uDC4E", -1, "Not helpful"]].forEach(function (def) {
                var btn = document.createElement("button");
                btn.type = "button";
                btn.className = "fb-btn";
                btn.textContent = def[1];
                btn.title = def[3];
                btn.setAttribute("aria-label", def[3]);
                btn.addEventListener("click", function () { sendFeedback(meta.chat_id, def[2], bar, btn); });
                bar.appendChild(btn);
            });
        }
        return bar;
    }

    async function sendFeedback(chatId, rating, bar, btn) {
        try {
            var res = await fetch("/feedback", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ chat_id: chatId, rating: rating })
            });
            var result = await res.json();
            if (!res.ok || !result.ok) throw new Error("feedback failed");
            bar.querySelectorAll(".fb-btn").forEach(function (b) { b.classList.remove("chosen"); });
            btn.classList.add("chosen");
            showToast("Thanks for the feedback!");
        } catch (e) {
            showToast("Could not save feedback.");
        }
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
        sessionId = makeSessionId();
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
        setStatus(value ? "Thinking..." : "Ready", value ? "searching" : "ready");
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
