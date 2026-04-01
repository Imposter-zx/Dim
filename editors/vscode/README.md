# Dim Language VS Code Extension

A VS Code extension providing syntax highlighting and language support for the Dim programming language.

## Features

- Syntax highlighting for `.dim` files
- Comment support (line `//` and block `/* */`)
- Bracket matching and auto-closing
- Smart indentation for blocks (fn, if, match, for, while, etc.)
- Number literal highlighting (decimal, hex, binary, octal)
- String literal highlighting

## Installation

### From Source

1. Navigate to the `editors/vscode` directory
2. Run `npm install` to install dependencies
3. Run `npm run package` to create the VSIX package
4. Install the VSIX file via VS Code: `Extensions > Install from VSIX...`

Or, for development:

1. Open the `editors/vscode` folder in VS Code
2. Press `F5` to launch the extension in a new window

### Manual Installation

Copy the `syntaxes/dim.tmLanguage.json` and `language-configuration/dim.json` files to your VS Code user settings:

1. Open VS Code settings (`Ctrl+,` or `Cmd+,`)
2. Add the language configuration path to `files.associations`
3. Copy the grammar to your user `syntaxes` folder

## Extension Settings

No settings required - works out of the box with `.dim` file extension.

## License

MIT