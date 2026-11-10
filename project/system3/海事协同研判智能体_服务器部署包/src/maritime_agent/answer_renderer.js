/* Render the small Markdown subset used by agent answers without inserting model HTML. */
(function () {
  "use strict";

  const inlineTokens = [
    { tag: "code", pattern: /`([^`\n]+)`/g },
    { tag: "strong", pattern: /\*\*([^\n]+?)\*\*/g },
    { tag: "strong", pattern: /__([^\n]+?)__/g },
    { tag: "em", pattern: /\*([^*\n]+?)\*/g },
    { tag: "a", pattern: /\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/gi },
    { tag: "citation", pattern: /【([^】\n]+)】/g },
  ];

  function appendInline(parent, input, options) {
    let position = 0;
    while (position < input.length) {
      let selected = null;
      for (const token of inlineTokens) {
        token.pattern.lastIndex = position;
        const match = token.pattern.exec(input);
        if (match && (!selected || match.index < selected.match.index)) {
          selected = { token, match };
        }
      }
      if (!selected) {
        parent.append(document.createTextNode(input.slice(position)));
        break;
      }
      if (selected.match.index > position) {
        parent.append(document.createTextNode(input.slice(position, selected.match.index)));
      }
      const { token, match } = selected;
      if (token.tag === "citation") {
        const number = options.citationNumbers?.get(match[1]);
        if (number) {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "citation";
          button.textContent = `[${number}]`;
          button.title = `查看证据 ${number}`;
          button.setAttribute("aria-label", `查看证据 ${number}`);
          button.addEventListener("click", () => options.onCitation?.(match[1]));
          parent.append(button);
        } else {
          parent.append(document.createTextNode(match[0]));
        }
        position = match.index + match[0].length;
        continue;
      }
      const element = document.createElement(token.tag);
      if (token.tag === "code") {
        element.textContent = match[1];
      } else if (token.tag === "a") {
        try {
          const url = new URL(match[2]);
          if (url.protocol !== "http:" && url.protocol !== "https:") throw new Error("Unsupported URL");
          element.href = url.href;
          element.target = "_blank";
          element.rel = "noopener noreferrer";
          element.textContent = match[1];
        } catch (_) {
          element.textContent = match[0];
        }
      } else {
        appendInline(element, match[1], options);
      }
      parent.append(element);
      position = match.index + match[0].length;
    }
  }

  function appendLines(parent, lines, options) {
    lines.forEach((line, index) => {
      if (index) parent.append(document.createElement("br"));
      appendInline(parent, line.trim(), options);
    });
  }

  function listMatch(line) {
    return /^(\s*)([-+*]|\d+[.)])\s+(.+)$/.exec(line);
  }

  function tableCells(line) {
    return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(cell => cell.trim());
  }

  function isTableStart(lines, index) {
    if (index + 1 >= lines.length || !lines[index].includes("|")) return false;
    const header = tableCells(lines[index]);
    const divider = tableCells(lines[index + 1]);
    return header.length > 1 && header.length === divider.length &&
      divider.every(cell => /^:?-{3,}:?$/.test(cell));
  }

  function startsBlock(lines, index) {
    const line = lines[index];
    return /^\s*(```|~~~|#{1,6}\s|>|---+\s*$)/.test(line) ||
      Boolean(listMatch(line)) || isTableStart(lines, index);
  }

  function renderAnswerMarkdown(target, markdown, options = {}) {
    const lines = String(markdown || "").replace(/\r\n?/g, "\n").split("\n");
    const content = document.createDocumentFragment();
    let index = 0;
    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) { index++; continue; }

      const fence = /^\s*(```|~~~)/.exec(line);
      if (fence) {
        const codeLines = [];
        index++;
        while (index < lines.length && !lines[index].trimStart().startsWith(fence[1])) {
          codeLines.push(lines[index++]);
        }
        if (index < lines.length) index++;
        const block = document.createElement("pre");
        const code = document.createElement("code");
        code.textContent = codeLines.join("\n");
        block.append(code);
        content.append(block);
        continue;
      }

      const heading = /^\s*(#{1,6})\s+(.+)$/.exec(line);
      if (heading) {
        const element = document.createElement("h" + Math.min(heading[1].length + 2, 6));
        appendInline(element, heading[2], options);
        content.append(element);
        index++;
        continue;
      }

      if (isTableStart(lines, index)) {
        const wrapper = document.createElement("div");
        wrapper.className = "answer-table-wrap";
        const table = document.createElement("table");
        const head = document.createElement("thead");
        const headRow = document.createElement("tr");
        tableCells(lines[index]).forEach(value => {
          const cell = document.createElement("th");
          appendInline(cell, value, options);
          headRow.append(cell);
        });
        head.append(headRow);
        table.append(head);
        index += 2;
        const body = document.createElement("tbody");
        while (index < lines.length && lines[index].trim() && lines[index].includes("|")) {
          const row = document.createElement("tr");
          tableCells(lines[index++]).forEach(value => {
            const cell = document.createElement("td");
            appendInline(cell, value, options);
            row.append(cell);
          });
          body.append(row);
        }
        table.append(body);
        wrapper.append(table);
        content.append(wrapper);
        continue;
      }

      const item = listMatch(line);
      if (item) {
        const ordered = /^\d/.test(item[2]);
        const list = document.createElement(ordered ? "ol" : "ul");
        if (ordered) list.start = Number.parseInt(item[2], 10);
        const indentation = item[1].length;
        while (index < lines.length) {
          const current = listMatch(lines[index]);
          if (!current || current[1].length !== indentation || /^\d/.test(current[2]) !== ordered) break;
          const element = document.createElement("li");
          appendInline(element, current[3], options);
          list.append(element);
          index++;
        }
        content.append(list);
        continue;
      }

      if (/^\s*>/.test(line)) {
        const quote = document.createElement("blockquote");
        const quoteLines = [];
        while (index < lines.length && /^\s*>/.test(lines[index])) {
          quoteLines.push(lines[index++].replace(/^\s*>\s?/, ""));
        }
        appendLines(quote, quoteLines, options);
        content.append(quote);
        continue;
      }

      if (/^\s*---+\s*$/.test(line)) {
        content.append(document.createElement("hr"));
        index++;
        continue;
      }

      const paragraph = document.createElement("p");
      const paragraphLines = [];
      while (index < lines.length && lines[index].trim() &&
             (paragraphLines.length === 0 || !startsBlock(lines, index))) {
        paragraphLines.push(lines[index++]);
      }
      appendLines(paragraph, paragraphLines, options);
      content.append(paragraph);
    }
    target.replaceChildren(content);
  }

  window.renderAnswerMarkdown = renderAnswerMarkdown;
})();
