function Math(el)
  el.text = el.text:gsub("{\\rm%s+([^{}]*)}", "\\mathrm{%1}"):gsub("{\\tt%s+([^{}]*)}", "\\mathtt{%1}"):gsub("{\\it%s+([^{}]*)}", "\\mathit{%1}")
  return el
end
