const copyButton = document.getElementById("copy-bibtex");
const bibtex = document.getElementById("bibtex");

if (copyButton && bibtex && navigator.clipboard) {
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bibtex.innerText.trim());
      const original = copyButton.textContent;
      copyButton.textContent = "Copied";
      window.setTimeout(() => {
        copyButton.textContent = original;
      }, 1400);
    } catch (error) {
      copyButton.textContent = "Copy failed";
    }
  });
}
