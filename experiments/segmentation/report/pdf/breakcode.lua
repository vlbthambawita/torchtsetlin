-- Let inline code break after _ , / = so long record IDs wrap inside table cells.
local function esc(s)
  s = s:gsub("\\", "\\textbackslash{}"):gsub("([#$%%&{}])", "\\%1")
  s = s:gsub("~", "\\textasciitilde{}"):gsub("%^", "\\textasciicircum{}")
  s = s:gsub("_", "\\_\\allowbreak{}"):gsub("([,/=])", "%1\\allowbreak{}")
  return s
end
function Code(el)
  return pandoc.RawInline("latex", "\\texttt{" .. esc(el.text) .. "}")
end
function Str(el)
  if #el.text > 12 and el.text:find("_") then
    return pandoc.RawInline("latex", esc(el.text))
  end
end
-- Short tables (<= 30 rows) become an unbreakable tabular: longtable strands a repeated header
-- at page breaks. Headings reserve room so they are never left alone at the bottom of a page.
local function rows(t)
  local n = 0
  for _, b in ipairs(t.bodies) do n = n + #b.body end
  return n
end
local function plain(t)
  local tex = pandoc.write(pandoc.Pandoc({ t }), "latex")
  tex = tex:gsub("\\endhead\n", "", 1)
  tex = tex:gsub("\\bottomrule\\noalign{}\n\\endlastfoot\n", "", 1)
  tex = tex:gsub("\\end{longtable}", "\\bottomrule\\noalign{}\n\\end{tabular}\\par\\medskip", 1)
  tex = tex:gsub("\\begin{longtable}%[%]", "\\par\\medskip\\noindent\\footnotesize\\begin{tabular}", 1)
  return pandoc.RawBlock("latex", tex)
end
local function need(n) return pandoc.RawBlock("latex", "\\needspace{" .. n .. "\\baselineskip}") end
function Blocks(bs)
  local out = {}
  for i, b in ipairs(bs) do
    if b.t == "Header" then
      local nx, n = bs[i + 1], 12
      if nx and nx.t == "Table" and rows(nx) <= 30 then n = rows(nx) + 8 end
      table.insert(out, need(n))
    end
    if b.t == "Table" and rows(b) <= 30 then
      table.insert(out, plain(b))
    else
      table.insert(out, b)
    end
  end
  return out
end
function Image(el)
  el.attributes.width = el.src:find("qualitative") and "85%" or "72%"
  return el
end
