(function () {
  const currentScript = document.currentScript;
  const assetBase = currentScript?.src ? new URL(".", currentScript.src).toString() : "./";
  const scriptOrigin = currentScript?.src ? new URL(currentScript.src).origin : "";
  const pageOrigin = window.location.protocol.startsWith("http") ? window.location.origin : "";
  const API_BASE = window.MIGTORG_CHAT_API_BASE || scriptOrigin || pageOrigin || "http://127.0.0.1:8000";
  ensureWidgetShell();
  const root = document.querySelector(".chat");
  const toggle = document.querySelector(".chat__toggle");
  const close = document.querySelector(".chat__close");
  const messages = document.querySelector(".chat__messages");
  const navigation = document.querySelector(".chat__navigation");
  const homeButton = document.querySelector(".chat__home");
  const form = document.querySelector(".chat__form");
  const input = document.querySelector(".chat__input");
  const ticketButton = document.querySelector(".chat__ticket-button");
  const ticketActions = document.querySelector(".chat__actions");
  const ticketForm = document.querySelector(".ticket-form");
  const quickButtons = document.querySelectorAll(".chat__quick button");
  const demoControls = document.querySelector(".demo__controls");
  const debugOutput = document.querySelector(".demo__debug");
  const resetSessionButton = document.querySelector(".demo__reset-session");
  const siteContext = window.MIGTORG_CHAT_CONTEXT || {};
  const trustedContextToken = window.MIGTORG_CHAT_TRUSTED_CONTEXT_TOKEN || null;
  let sessionId = getSessionId();
  let chatGeneration = 0;
  let latestTicketContext = null;
  let latestMessageId = null;

  if (!root || !toggle || !close || !messages || !navigation || !homeButton || !form || !input || !ticketButton || !ticketForm) {
    console.warn("MIGTORG chat widget: shell markup is incomplete.");
    return;
  }

  function ensureWidgetShell() {
    if (document.querySelector(".chat")) return;
    ensureStylesheet();
    const wrapper = document.createElement("div");
    wrapper.className = "chat";
    wrapper.dataset.open = "false";
    wrapper.innerHTML = `
      <button class="chat__toggle" type="button" aria-label="Открыть поддержку MIGTORG">
        <svg class="chat__toggle-icon" viewBox="0 0 24 24" width="24" height="24" aria-hidden="true" focusable="false">
          <path d="M6 4h12a4 4 0 0 1 4 4v6a4 4 0 0 1-4 4h-6l-5 3v-3H6a4 4 0 0 1-4-4V8a4 4 0 0 1 4-4Z" />
        </svg>
        <span class="chat__toggle-text">Поддержка MIGTORG</span>
      </button>

      <section class="chat__panel" aria-label="Поддержка MIGTORG">
        <header class="chat__header">
          <div class="chat__brand">
            <span class="chat__logo" aria-hidden="true">M</span>
            <div>
              <strong>Поддержка MIGTORG</strong>
              <span>Лоты, торги, оплата и документы</span>
            </div>
          </div>
          <button class="chat__close" type="button" aria-label="Закрыть чат">×</button>
        </header>

        <div class="chat__messages" aria-live="polite">
          <div class="message message--bot message--welcome">Опишите проблему своими словами — я помогу разобраться или подготовлю обращение в поддержку.</div>
          <div class="chat__quick-intro">
            <strong>Частые ситуации</strong>
          </div>
          <div class="chat__quick" aria-label="Частые ситуации">
            <button type="button" data-message="Лот не передают">Лот не передают</button>
            <button type="button" data-message="Не могу сделать ставку">Не могу сделать ставку</button>
            <button type="button" data-message="Оплатил тариф, доступа нет">Оплатил тариф, доступа нет</button>
            <button type="button" data-message="Вопрос по штрафу или депозиту">Вопрос по штрафу или депозиту</button>
          </div>
        </div>

        <nav class="chat__navigation" aria-label="Навигация по чату" hidden>
          <button class="chat__home" type="button" aria-label="Вернуться в начало и начать новый диалог">В начало</button>
        </nav>

        <form class="chat__form">
          <input class="chat__input" name="message" autocomplete="off" placeholder="Напишите вопрос своими словами" />
          <button class="chat__send" type="submit" aria-label="Отправить сообщение">Отправить</button>
        </form>

        <div class="chat__actions" hidden>
          <button class="chat__ticket-button" type="button">Создать обращение</button>
        </div>

        <form class="ticket-form" hidden>
          <label>
            <span>Контакт</span>
            <input name="contact" placeholder="Телефон или email" />
          </label>
          <label>
            <span>Тема</span>
            <select name="topic">
              <option value="Обращение в поддержку">Обращение в поддержку</option>
              <option value="Проверка платежа">Проверка платежа</option>
              <option value="Возврат">Возврат</option>
              <option value="Вопрос по начислению штрафа">Вопрос по начислению штрафа</option>
              <option value="Отказ от лота">Отказ от лота</option>
              <option value="Вопрос по лоту">Вопрос по лоту</option>
              <option value="Получение автомобиля">Получение автомобиля</option>
              <option value="Вопрос по тарифу">Вопрос по тарифу</option>
              <option value="Закрывающие документы по сделке">Закрывающие документы по сделке</option>
              <option value="Обратная связь">Обратная связь</option>
              <option value="Другая тема">Другая тема</option>
            </select>
          </label>
          <label class="ticket-form__custom-topic" hidden>
            <span>Своя тема</span>
            <input name="custom_topic" placeholder="Коротко укажите тему" />
          </label>
          <label class="ticket-form__description">
            <span>Описание</span>
            <textarea name="description" placeholder="Опишите ситуацию, номер лота или платежа"></textarea>
          </label>
          <button class="ticket-form__clear-description" type="button">Очистить и написать своё</button>
          <button type="submit">Создать обращение</button>
        </form>
      </section>
    `;
    document.body.appendChild(wrapper);
  }

  function ensureStylesheet() {
    if (document.getElementById("migtorg-chat-widget-style")) return;
    const link = document.createElement("link");
    link.id = "migtorg-chat-widget-style";
    link.rel = "stylesheet";
    link.href = new URL("style.css?v=20260905-guided-navigation", assetBase).toString();
    document.head.appendChild(link);
  }

  function getSessionId() {
    const existing = localStorage.getItem("migtorg_chat_session_id");
    if (existing) return existing;
    const created = crypto.randomUUID ? crypto.randomUUID() : String(Date.now());
    localStorage.setItem("migtorg_chat_session_id", created);
    return created;
  }

  const homeMessageNodes = Array.from(messages.children);

  function setHomeVisible(isVisible) {
    navigation.hidden = !isVisible;
  }

  setHomeVisible(false);

  function returnToStart() {
    chatGeneration += 1;
    localStorage.removeItem("migtorg_chat_session_id");
    sessionId = getSessionId();
    latestTicketContext = null;
    latestMessageId = null;
    messages.replaceChildren(...homeMessageNodes);
    quickButtons.forEach((button) => {
      button.disabled = false;
    });
    form.reset();
    ticketForm.reset();
    resetTicketFormState();
    setTicketFormOpen(false);
    setTicketButtonVisible(false);
    setHomeVisible(false);
    if (debugOutput) debugOutput.textContent = "";
    input.focus();
    loadStartExperience();
  }

  function context(extra) {
    return Object.assign({}, siteContext, demoContext(), extra || {}, { session_id: sessionId });
  }

  function demoContext() {
    if (!demoControls) return {};
    const isAuthorized = demoControls.querySelector('[name="is_authorized"]')?.value === "true";
    return {
      is_authorized: isAuthorized,
      user_id: valueOf("user_id"),
      page_type: valueOf("page_type") || "public_site",
      lot_id: valueOf("lot_id"),
      user_email: valueOf("user_email"),
      user_phone: valueOf("user_phone"),
    };
  }

  function valueOf(name) {
    const node = demoControls?.querySelector(`[name="${name}"]`);
    const value = node ? String(node.value || "").trim() : "";
    return value || null;
  }

  function updateDebug(data) {
    if (!debugOutput) return;
    debugOutput.textContent = JSON.stringify(
      {
        session_id: data.session_id || sessionId,
        role: data.role,
        intent: data.intent,
        needs_ticket: data.needs_ticket,
        ticket_id: data.ticket_id,
        attachments: data.attachments || [],
        template_links: data.template_links || [],
        safety_flags: data.safety_flags || data.safety_categories || [],
        model_used: data.model_used || "mock",
        scenario_id: data.scenario_id || null,
        resolution: data.resolution || null,
        experience_variant: data.experience_variant || null,
        navigation_version: data.navigation_version || null,
        navigation_node_id: data.navigation_node_id || null,
      },
      null,
      2,
    );
  }

  function setMessageText(node, text) {
    const value = String(text || "");
    const urlPattern = /https?:\/\/[^\s]+/g;
    let cursor = 0;

    for (const match of value.matchAll(urlPattern)) {
      const rawUrl = match[0];
      const url = rawUrl.replace(/[),.;!?]+$/, "");
      const start = match.index || 0;
      node.appendChild(document.createTextNode(value.slice(cursor, start)));

      const link = document.createElement("a");
      link.className = "message__inline-link";
      link.href = url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = url;
      node.appendChild(link);

      const trailing = rawUrl.slice(url.length);
      if (trailing) node.appendChild(document.createTextNode(trailing));
      cursor = start + rawUrl.length;
    }

    node.appendChild(document.createTextNode(value.slice(cursor)));
  }

  function appendMessage(text, owner, modifier) {
    const node = document.createElement("div");
    node.className = `message message--${owner}${modifier ? ` message--${modifier}` : ""}`;
    setMessageText(node, text);
    messages.appendChild(node);
    messages.scrollTop = messages.scrollHeight;
    return node;
  }

  function appendMessageActions(node, actions, modifier) {
    if (!node || !Array.isArray(actions) || !actions.length) return;
    const actionsNode = document.createElement("div");
    actionsNode.className = `message__actions${modifier ? ` message__actions--${modifier}` : ""}`;
    actions.forEach((action) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = action.label;
      button.addEventListener("click", action.onClick);
      actionsNode.appendChild(button);
    });
    node.appendChild(actionsNode);
    messages.scrollTop = messages.scrollHeight;
  }

  function openTicketFlow(description, intent, options) {
    rememberTicketContext(description, intent, options);
    prefillTicketForm(latestTicketContext);
    setTicketFormOpen(true);
  }

  function rememberTicketContext(description, intent, options) {
    if (!description || !intent || ["unknown", "safety", "prohibited"].includes(intent)) return;
    latestTicketContext = Object.assign({ description, intent }, options || {});
  }

  function focusForClarification() {
    setTicketFormOpen(false);
    input.focus();
  }

  function appendNotHelpfulAction(node, description, intent) {
    appendMessageActions(node, [
      {
        label: "Не помогло",
        onClick: async (event) => {
          const generationAtClick = chatGeneration;
          event.currentTarget.disabled = true;
          try {
            await postJson("/api/chat/feedback", {
              session_id: sessionId,
              rating: 1,
              comment: "Не помогло",
              message_id: latestMessageId,
            });
          } catch (error) {
            // Feedback is helpful for analytics, but the user flow should continue even if it fails.
          }
          if (generationAtClick !== chatGeneration) return;
          const prompt = appendMessage("Передать вопрос сотрудникам?", "system");
          appendMessageActions(prompt, [
            {
              label: "Создать обращение",
              onClick: () => openTicketFlow(description, intent),
            },
            {
              label: "Попробовать уточнить",
              onClick: focusForClarification,
            },
          ]);
        },
      },
    ]);
  }

  function appendClarifyingOptions(node, options) {
    if (!Array.isArray(options) || !options.length) return;
    appendMessageActions(
      node,
      options.map((option) => ({
        label: option,
        onClick: (event) => {
          event.currentTarget.parentElement
            ?.querySelectorAll("button")
            .forEach((button) => {
              button.disabled = true;
            });
          sendMessage(option);
        },
      })),
      "clarifying",
    );
  }

  function appendStructuredActions(node, actions, message, intent) {
    if (!Array.isArray(actions) || !actions.length) return false;
    const modifier = actions.some((action) => action.type === "guided_choice")
      ? "guided"
      : "clarifying";
    appendMessageActions(
      node,
      actions.map((action) => ({
        label: action.label,
        onClick: (event) => {
          if (action.type === "answer" && action.payload?.message) {
            event.currentTarget.parentElement
              ?.querySelectorAll("button")
              .forEach((button) => {
                button.disabled = true;
              });
            sendMessage(action.payload.message);
            return;
          }
          if (action.type === "navigate" && action.payload?.url) {
            window.location.assign(action.payload.url);
            return;
          }
          if (["open_ticket", "handoff", "request_callback"].includes(action.type)) {
            openTicketFlow(message, intent, {
              scenario_id: action.scenario_id || null,
            });
            return;
          }
          event.currentTarget.parentElement
            ?.querySelectorAll("button")
            .forEach((button) => {
              button.disabled = true;
            });
          const pending = sendMessage(action.label, action.id);
          if (action.type === "guided_choice" && action.payload?.kind === "free_text") {
            pending.finally(() => input.focus());
          }
        },
      })),
      modifier,
    );
    return true;
  }

  function documentLabelFromUrl(url) {
    try {
      const pathname = new URL(url, API_BASE).pathname;
      const filename = decodeURIComponent(pathname.split("/").filter(Boolean).pop() || "");
      return filename || "Скачать документ";
    } catch (error) {
      return "Скачать документ";
    }
  }

  function normalizeDocumentLinks(data) {
    const result = [];
    const seen = new Set();

    function add(label, url) {
      if (!url) return;
      const key = String(url);
      if (seen.has(key)) return;
      seen.add(key);
      result.push({
        label: label || documentLabelFromUrl(url),
        url,
      });
    }

    (data.template_links || []).forEach((link) => add(link.label, link.url));
    (data.attachments || []).forEach((attachment) => {
      if (typeof attachment === "string") {
        add(documentLabelFromUrl(attachment), attachment);
        return;
      }
      if (attachment && typeof attachment === "object") {
        add(attachment.label || attachment.name, attachment.url || attachment.href);
      }
    });

    return result;
  }

  function appendTemplateLinks(node, links) {
    if (!Array.isArray(links) || !links.length) return;
    const list = document.createElement("div");
    list.className = "message__links";
    const title = document.createElement("span");
    title.className = "message__links-title";
    title.textContent = "Документ для скачивания:";
    list.appendChild(title);
    links.forEach((link) => {
      const anchor = document.createElement("a");
      anchor.href = new URL(link.url, API_BASE).toString();
      anchor.target = "_blank";
      anchor.rel = "noopener";
      anchor.textContent = link.label || "Скачать документ";
      list.appendChild(anchor);
    });
    node.appendChild(list);
    messages.scrollTop = messages.scrollHeight;
  }

  async function postJson(path, payload) {
    const response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  async function loadStartExperience() {
    const generationAtStart = chatGeneration;
    try {
      const data = await postJson("/api/chat/start", {
        session_id: sessionId,
        context: context(),
      });
      if (generationAtStart !== chatGeneration) return;
      latestMessageId = data.message_id || null;
      messages.replaceChildren();
      const welcome = appendMessage(data.answer, "bot", "welcome");
      appendStructuredActions(welcome, data.actions || [], data.answer, data.intent);
      updateDebug(data);
    } catch (error) {
      // The static welcome and free-text input remain usable when start loading fails.
    }
  }

  async function sendMessage(text, selectedActionId) {
    const message = text.trim();
    if (!message) return;
    const generationAtSend = chatGeneration;
    setHomeVisible(true);
    setTicketButtonVisible(false);
    appendMessage(message, "user");
    const loading = appendMessage("Разбираю ситуацию...", "bot", "loading");
    try {
      const data = await postJson("/api/chat/message", {
        message,
        session_id: sessionId,
        context: context(),
        selected_action_id: selectedActionId || null,
        conversation_turn_id: latestMessageId,
        trusted_context_token: trustedContextToken,
      });
      if (generationAtSend !== chatGeneration) return;
      latestMessageId = data.message_id || null;
      loading.classList.remove("message--loading");
      loading.replaceChildren();
      setMessageText(loading, data.answer);
      appendTemplateLinks(loading, normalizeDocumentLinks(data));
      const hasStructuredActions = appendStructuredActions(
        loading,
        data.actions || [],
        message,
        data.intent,
      );
      if (data.action === "clarify" && !hasStructuredActions) {
        appendClarifyingOptions(loading, data.clarifying_options || []);
      } else if (!hasStructuredActions) {
        rememberTicketContext(message, data.intent);
        appendNotHelpfulAction(loading, message, data.intent);
      }
      updateDebug(data);
      if (data.needs_ticket && data.ticket_id) {
        appendMessage(`Обращение создано. Номер: ${data.ticket_id}.`, "system");
      } else if (data.needs_ticket) {
        setTicketButtonVisible(true);
      }
    } catch (error) {
      if (generationAtSend !== chatGeneration) return;
      loading.classList.remove("message--loading");
      loading.textContent = "Не удалось отправить сообщение. Проверьте подключение к backend.";
    }
  }

  toggle.addEventListener("click", () => {
    root.dataset.open = "true";
    input.focus();
  });

  close.addEventListener("click", () => {
    root.dataset.open = "false";
  });

  homeButton.addEventListener("click", returnToStart);

  quickButtons.forEach((button) => {
    button.addEventListener("click", () => {
      sendMessage(button.dataset.message || button.textContent || "");
    });
  });

  loadStartExperience();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = input.value;
    input.value = "";
    await sendMessage(text);
  });

  ticketButton.addEventListener("click", () => {
    const willOpen = ticketForm.hidden;
    if (willOpen) prefillTicketForm(latestTicketContext);
    setTicketFormOpen(willOpen);
  });

  function setTicketButtonVisible(isVisible) {
    if (!ticketActions) return;
    ticketActions.hidden = !isVisible;
  }

  function setTicketFormOpen(isOpen) {
    ticketForm.hidden = !isOpen;
    root.dataset.formOpen = isOpen ? "true" : "false";
    ticketButton.textContent = isOpen ? "Скрыть форму" : "Создать обращение";
    if (isOpen) {
      messages.scrollTop = Math.max(0, messages.scrollHeight - messages.clientHeight);
    }
  }

  function prefillTicketForm(ticketContext) {
    const currentContext = context();
    const contactInput = ticketForm.querySelector('[name="contact"]');
    const topicSelect = ticketForm.querySelector('[name="topic"]');
    const descriptionInput = ticketForm.querySelector('[name="description"]');
    if (contactInput && !contactInput.value) {
      contactInput.value = currentContext.user_email || currentContext.user_phone || "";
    }
    if (topicSelect && topicSelect.dataset.userEdited !== "true") {
      topicSelect.value = ticketContext
        ? topicByIntent(ticketContext.intent, ticketContext.description)
        : "Обращение в поддержку";
      updateCustomTopicVisibility(false);
    }
    if (descriptionInput && descriptionInput.dataset.userEdited !== "true") {
      descriptionInput.value = ticketContext
        ? descriptionByIntent(ticketContext.description, ticketContext.intent, currentContext)
        : "";
    }
  }

  function isDocumentRequest(message) {
    return /(документ|доки|закрывающ|счет.?фактур|счёт.?фактур|бухгалтер)/i.test(String(message || ""));
  }

  function topicByIntent(intent, message) {
    if (isDocumentRequest(message)) return "Закрывающие документы по сделке";
    const topics = {
      payment: "Проверка платежа",
      refund: "Возврат",
      penalty: "Вопрос по начислению штрафа",
      refusal: "Отказ от лота",
      lot: "Вопрос по лоту",
      support: "Обращение в поддержку",
      feedback: "Обратная связь",
      pickup: "Получение автомобиля",
      tariffs: "Вопрос по тарифу",
    };
    return topics[intent] || "Обращение в поддержку";
  }

  function descriptionByIntent(message, intent, currentContext) {
    const lotNumber = currentContext.lot_id || "…";
    if (isDocumentRequest(message)) {
      return `Закрывающие документы по лоту № ${lotNumber}: ${message}`;
    }
    const prefixes = {
      payment: "Вопрос по платежу",
      refund: "Вопрос по возврату",
      penalty: `Начислен штраф по лоту № ${lotNumber}`,
      refusal: `Вопрос по отказу от лота № ${lotNumber}`,
      lot: `Вопрос по лоту № ${lotNumber}`,
      pickup: `Вопрос по получению лота № ${lotNumber}`,
      tariffs: "Вопрос по тарифу",
    };
    const prefix = prefixes[intent] || "Вопрос пользователя";
    return `${prefix}: ${message}`;
  }

  function updateCustomTopicVisibility(shouldFocus) {
    const topicSelect = ticketForm.querySelector('[name="topic"]');
    const customTopicRow = ticketForm.querySelector(".ticket-form__custom-topic");
    const customTopicInput = ticketForm.querySelector('[name="custom_topic"]');
    const isCustom = topicSelect?.value === "Другая тема";
    if (customTopicRow) customTopicRow.hidden = !isCustom;
    if (customTopicInput) customTopicInput.required = isCustom;
    if (isCustom && shouldFocus) customTopicInput?.focus();
  }

  function resetTicketFormState() {
    for (const field of ticketForm.querySelectorAll('[data-user-edited="true"]')) {
      delete field.dataset.userEdited;
    }
    updateCustomTopicVisibility(false);
  }

  const topicSelect = ticketForm.querySelector('[name="topic"]');
  const descriptionInput = ticketForm.querySelector('[name="description"]');
  const clearDescriptionButton = ticketForm.querySelector(".ticket-form__clear-description");

  topicSelect?.addEventListener("change", () => {
    topicSelect.dataset.userEdited = "true";
    updateCustomTopicVisibility(true);
  });

  descriptionInput?.addEventListener("input", () => {
    descriptionInput.dataset.userEdited = "true";
  });

  clearDescriptionButton?.addEventListener("click", () => {
    if (!descriptionInput) return;
    descriptionInput.value = "";
    descriptionInput.dataset.userEdited = "true";
    descriptionInput.focus();
  });

  ticketForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const generationAtSubmit = chatGeneration;
    const formData = new FormData(ticketForm);
    const description = String(formData.get("description") || "").trim();
    const contact = String(formData.get("contact") || "").trim() || context().user_email || context().user_phone || "";
    const selectedTopic = String(formData.get("topic") || "Обращение в поддержку");
    const customTopic = String(formData.get("custom_topic") || "").trim();
    const topic = selectedTopic === "Другая тема" ? customTopic : selectedTopic;
    if (!contact) {
      appendMessage("Укажите, пожалуйста, телефон или email, чтобы сотрудник мог связаться с вами.", "system");
      return;
    }
    if (!description) {
      appendMessage("Опишите, пожалуйста, ситуацию для обращения.", "system");
      return;
    }
    if (!topic) {
      appendMessage("Укажите, пожалуйста, тему обращения.", "system");
      ticketForm.querySelector('[name="custom_topic"]')?.focus();
      return;
    }
    try {
      const data = await postJson("/api/chat/ticket", {
        session_id: sessionId,
        context: context(),
        contact,
        topic,
        description,
        category: "support",
        scenario_id: latestTicketContext?.scenario_id || null,
        source_message_id: latestMessageId,
        trusted_context_token: trustedContextToken,
      });
      if (generationAtSubmit !== chatGeneration) return;
      updateDebug({
        session_id: sessionId,
        role: context().is_authorized ? "authorized" : "guest",
        intent: "manual_ticket",
        needs_ticket: true,
        ticket_id: data.ticket_id,
        attachments: [],
        safety_flags: [],
        model_used: "mock",
      });
      const deliveryLabels = {
        saved: "Отправка ещё не выполнена.",
        pending: "Обращение ожидает отправки.",
        sending: "Выполняется отправка обращения.",
        accepted: "Почтовый сервер поддержки принял обращение.",
        failed: "Не удалось отправить обращение.",
        unknown: "Результат отправки требует проверки.",
      };
      const deliveryText = deliveryLabels[data.delivery?.state] || "Статус отправки пока не подтверждён.";
      appendMessage(`Обращение создано. Номер: ${data.ticket_id}. ${deliveryText}`, "system");
      ticketForm.reset();
      resetTicketFormState();
      setTicketFormOpen(false);
      setTicketButtonVisible(false);
    } catch (error) {
      if (generationAtSubmit !== chatGeneration) return;
      appendMessage("Не удалось создать обращение. Попробуйте еще раз.", "system");
    }
  });

  resetSessionButton?.addEventListener("click", () => {
    localStorage.removeItem("migtorg_chat_session_id");
    window.location.reload();
  });
})();
