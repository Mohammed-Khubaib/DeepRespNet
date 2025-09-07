---
icon: material/brain
hide:
  # - navigation
  - toc
---
# DeepRespNet
## Web Application

<video controls width="600">
  <source src="assets/App.mov">
  Your browser does not support the video tag.
</video>

```python hl_lines="2 3"
def hello_world():
    print("Hello, World!")
    return True
```


```json
{
  "navigation": {
    "groups": [
      {
        "group": "Getting Started",
		"icon": "play",
        "pages": [
          "quickstart",
          {
            "group": "Editing",
			"icon": "pencil",
            "pages": [
				"installation",
				"editor",
				{
					"group": "Nested group",
					"icon": "code",
					"pages": [
						"navigation",
						"code"
					]
				}
			]
          }
        ]
      },
      {
        "group": "Writing Content",
 		"icon": "notebook-text",
        "tag": "NEW",
        "pages": ["writing-content/page", "writing-content/text"]
      }
    ]
  }
}
```




This are some of the markdown extension that I have, do i need to change anything, to make syntax highlighting work? ryt now, it isn't, any theme that works for vs code will work:
# Markdown Extensions
markdown_extensions:
  - abbr
  - admonition
  - pymdownx.highlight:
      anchor_linenums: true
      line_spans: __span
      pygments_lang_class: true
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.superfences
  
  - pymdownx.details
  - attr_list
  - def_list
  - footnotes
  - md_in_html
  
  # - toc:
  #     permalink: true
  #     toc_depth: 4
  - toc:
      anchorlink: true
      anchorlink_class: "toclink"
      # toc_depth: 1
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.betterem:
      smart_enable: all
  - pymdownx.caret
  - pymdownx.emoji:
      # emoji_generator: !!python/name:materialx.emoji.to_svg
      # emoji_index: !!python/name:materialx.emoji.twemoji
  - pymdownx.keys
  - pymdownx.magiclink:
      # repo_url_shorthand: true
      # user: your-username
      # repo: your-repo-name
  - pymdownx.superfences:
  - markdown.extensions.attr_list:
  - pymdownx.keys:
  - pymdownx.tasklist:
      custom_checkbox: true
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.mark
  - pymdownx.smartsymbols
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.tasklist:
      custom_checkbox: true
  - pymdownx.tilde