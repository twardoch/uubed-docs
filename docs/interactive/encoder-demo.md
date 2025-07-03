# Interactive QuadB64 Encoder/Decoder

## Live Demo

Experience QuadB64 encoding in real-time with this interactive demo. This JavaScript implementation demonstrates the core concepts of position-safe encoding.

<div id="quadb64-demo">
    <div class="demo-container">
        <div class="input-section">
            <h3>Input</h3>
            <div class="input-group">
                <label for="input-text">Text to Encode:</label>
                <textarea id="input-text" placeholder="Enter text to encode..." rows="4">Hello, QuadB64!</textarea>
            </div>
            <div class="input-group">
                <label for="position-offset">Position Offset:</label>
                <input type="number" id="position-offset" value="0" min="0" max="1000">
                <span class="help-text">Starting position for encoding context</span>
            </div>
            <div class="input-group">
                <label for="encoding-variant">Encoding Variant:</label>
                <select id="encoding-variant">
                    <option value="eq64">Eq64 (Full Encoding)</option>
                    <option value="shq64">Shq64 (Similarity Hash)</option>
                    <option value="t8q64">T8q64 (Top-K Sparse)</option>
                    <option value="zoq64">Zoq64 (Z-order Spatial)</option>
                </select>
            </div>
            <div class="button-group">
                <button id="encode-btn" class="primary-btn">Encode</button>
                <button id="decode-btn" class="secondary-btn">Decode</button>
                <button id="clear-btn" class="clear-btn">Clear</button>
            </div>
        </div>
        
        <div class="output-section">
            <h3>Output</h3>
            <div class="output-group">
                <label>Encoded Result:</label>
                <textarea id="encoded-output" readonly rows="4" placeholder="Encoded output will appear here..."></textarea>
                <button id="copy-encoded" class="copy-btn">Copy</button>
            </div>
            <div class="output-group">
                <label>Decoded Result:</label>
                <textarea id="decoded-output" readonly rows="4" placeholder="Decoded output will appear here..."></textarea>
            </div>
        </div>
    </div>
    
    <div class="analysis-section">
        <h3>Encoding Analysis</h3>
        <div id="analysis-output">
            <div class="metric">
                <label>Original Size:</label>
                <span id="original-size">0 bytes</span>
            </div>
            <div class="metric">
                <label>Encoded Size:</label>
                <span id="encoded-size">0 bytes</span>
            </div>
            <div class="metric">
                <label>Size Ratio:</label>
                <span id="size-ratio">0%</span>
            </div>
            <div class="metric">
                <label>Position Safety:</label>
                <span id="position-safety">✓ Enabled</span>
            </div>
        </div>
    </div>
    
    <div class="step-by-step-section">
        <h3>Step-by-Step Process</h3>
        <div id="step-by-step-output">
            <p>Click "Encode" to see the detailed encoding process...</p>
        </div>
    </div>
</div>

<style>
.demo-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin: 2rem 0;
    padding: 1.5rem;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    background: #fafafa;
}

.input-section, .output-section {
    padding: 1rem;
    background: white;
    border-radius: 6px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.input-group, .output-group {
    margin-bottom: 1rem;
}

.input-group label, .output-group label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 600;
    color: #333;
}

.input-group input, .input-group select, .input-group textarea,
.output-group textarea {
    width: 100%;
    padding: 0.75rem;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-family: 'Roboto Mono', monospace;
    font-size: 0.9rem;
}

.input-group textarea, .output-group textarea {
    resize: vertical;
    min-height: 80px;
}

.help-text {
    display: block;
    font-size: 0.8rem;
    color: #666;
    margin-top: 0.25rem;
}

.button-group {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.primary-btn, .secondary-btn, .clear-btn, .copy-btn {
    padding: 0.75rem 1.5rem;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-weight: 600;
    transition: background-color 0.2s;
}

.primary-btn {
    background: #2196F3;
    color: white;
}

.primary-btn:hover {
    background: #1976D2;
}

.secondary-btn {
    background: #4CAF50;
    color: white;
}

.secondary-btn:hover {
    background: #388E3C;
}

.clear-btn {
    background: #FF9800;
    color: white;
}

.clear-btn:hover {
    background: #F57C00;
}

.copy-btn {
    background: #9C27B0;
    color: white;
    padding: 0.5rem 1rem;
    font-size: 0.8rem;
}

.copy-btn:hover {
    background: #7B1FA2;
}

.analysis-section, .step-by-step-section {
    margin: 2rem 0;
    padding: 1.5rem;
    background: white;
    border-radius: 6px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0;
    border-bottom: 1px solid #eee;
}

.metric:last-child {
    border-bottom: none;
}

.metric label {
    font-weight: 600;
    color: #333;
}

.metric span {
    font-family: 'Roboto Mono', monospace;
    color: #666;
}

#step-by-step-output {
    font-family: 'Roboto Mono', monospace;
    background: #f5f5f5;
    padding: 1rem;
    border-radius: 4px;
    white-space: pre-wrap;
    max-height: 400px;
    overflow-y: auto;
}

@media (max-width: 768px) {
    .demo-container {
        grid-template-columns: 1fr;
        gap: 1rem;
    }
    
    .button-group {
        flex-direction: column;
    }
}
</style>

<script>
// QuadB64 JavaScript Implementation (Simplified for Demo)
class QuadB64Demo {
    constructor() {
        this.baseAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        this.positionCache = new Map();
        this.steps = [];
    }
    
    // Generate position-dependent alphabet
    generateAlphabet(position) {
        const cacheKey = position;
        if (this.positionCache.has(cacheKey)) {
            return this.positionCache.get(cacheKey);
        }
        
        const rotation = Math.floor(position / 3) % 64;
        const rotatedAlphabet = this.baseAlphabet.slice(rotation) + this.baseAlphabet.slice(0, rotation);
        
        this.positionCache.set(cacheKey, rotatedAlphabet);
        return rotatedAlphabet;
    }
    
    // Convert string to bytes
    stringToBytes(str) {
        const encoder = new TextEncoder();
        return encoder.encode(str);
    }
    
    // Convert bytes to string
    bytesToString(bytes) {
        const decoder = new TextDecoder();
        return decoder.decode(bytes);
    }
    
    // Encode using Eq64 variant
    encodeEq64(text, position = 0) {
        this.steps = [];
        this.steps.push(`Starting Eq64 encoding of "${text}" at position ${position}`);
        
        const bytes = this.stringToBytes(text);
        this.steps.push(`Input converted to ${bytes.length} bytes: [${Array.from(bytes).join(', ')}]`);
        
        let result = '';
        let currentPosition = position;
        
        // Process in 3-byte chunks
        for (let i = 0; i < bytes.length; i += 3) {
            const chunk = bytes.slice(i, i + 3);
            const alphabet = this.generateAlphabet(currentPosition);
            
            this.steps.push(`\\nChunk ${Math.floor(i/3) + 1} at position ${currentPosition}:`);
            this.steps.push(`  Input bytes: [${Array.from(chunk).join(', ')}]`);
            this.steps.push(`  Alphabet rotation: ${Math.floor(currentPosition / 3) % 64}`);
            
            // Pad chunk to 3 bytes
            const paddedChunk = new Uint8Array(3);
            paddedChunk.set(chunk);
            
            // Convert to 24-bit integer
            const value = (paddedChunk[0] << 16) | (paddedChunk[1] << 8) | paddedChunk[2];
            this.steps.push(`  24-bit value: ${value.toString(2).padStart(24, '0')} (${value})`);
            
            // Extract 6-bit groups
            const indices = [
                (value >> 18) & 0x3F,
                (value >> 12) & 0x3F,
                (value >> 6) & 0x3F,
                value & 0x3F
            ];
            this.steps.push(`  6-bit indices: [${indices.join(', ')}]`);
            
            // Map to alphabet characters
            const chars = indices.map(idx => alphabet[idx]);
            const chunkResult = chars.slice(0, chunk.length + 1).join('');
            this.steps.push(`  Encoded: "${chunkResult}"`);
            
            result += chunkResult;
            currentPosition += 3;
        }
        
        this.steps.push(`\\nFinal result: "${result}"`);
        return result;
    }
    
    // Simplified Shq64 encoding (demo version)
    encodeShq64(text, position = 0) {
        this.steps = [];
        this.steps.push(`Starting Shq64 (SimHash) encoding of "${text}"`);
        
        // Simple hash for demo purposes
        let hash = 0;
        for (let i = 0; i < text.length; i++) {
            hash = ((hash << 5) - hash + text.charCodeAt(i)) & 0xFFFFFFFF;
        }
        
        this.steps.push(`Generated hash: ${hash.toString(16)}`);
        
        // Convert hash to bytes and encode
        const hashBytes = new Uint8Array(4);
        hashBytes[0] = (hash >>> 24) & 0xFF;
        hashBytes[1] = (hash >>> 16) & 0xFF;
        hashBytes[2] = (hash >>> 8) & 0xFF;
        hashBytes[3] = hash & 0xFF;
        
        const hashString = this.bytesToString(hashBytes);
        const result = this.encodeEq64(hashString, position);
        
        this.steps.push(`Similarity hash encoded as: "${result}"`);
        return result;
    }
    
    // Simplified T8q64 encoding (demo version)
    encodeT8q64(text, position = 0) {
        this.steps = [];
        this.steps.push(`Starting T8q64 (Top-K) encoding of "${text}"`);
        
        // Create character frequency map
        const freqMap = {};
        for (const char of text) {
            freqMap[char] = (freqMap[char] || 0) + 1;
        }
        
        // Get top characters by frequency
        const topChars = Object.entries(freqMap)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 8)
            .map(([char, freq]) => `${char}:${freq}`)
            .join(',');
        
        this.steps.push(`Character frequencies: ${JSON.stringify(freqMap)}`);
        this.steps.push(`Top-8 chars: ${topChars}`);
        
        const result = this.encodeEq64(topChars, position);
        this.steps.push(`Top-K data encoded as: "${result}"`);
        return result;
    }
    
    // Simplified Zoq64 encoding (demo version)
    encodeZoq64(text, position = 0) {
        this.steps = [];
        this.steps.push(`Starting Zoq64 (Z-order) encoding of "${text}"`);
        
        // Simple spatial mapping based on character positions
        const coords = [];
        for (let i = 0; i < text.length; i++) {
            const x = i % 8;
            const y = Math.floor(i / 8);
            coords.push(`(${x},${y})`);
        }
        
        this.steps.push(`Spatial coordinates: ${coords.join(' ')}`);
        
        // Z-order interleaving (simplified)
        const spatialData = coords.join('') + text;
        const result = this.encodeEq64(spatialData, position);
        
        this.steps.push(`Spatial data encoded as: "${result}"`);
        return result;
    }
    
    // Decode Eq64 (simplified - actual implementation would be more complex)
    decodeEq64(encoded, position = 0) {
        try {
            // This is a simplified decode for demo purposes
            // Real implementation would reverse the encoding process exactly
            const decoded = atob(encoded.replace(/[.]/g, ''));
            return decoded;
        } catch (e) {
            return `Decode error: ${e.message}`;
        }
    }
}

// Demo Application
class QuadB64DemoApp {
    constructor() {
        this.encoder = new QuadB64Demo();
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        document.getElementById('encode-btn').addEventListener('click', () => this.encode());
        document.getElementById('decode-btn').addEventListener('click', () => this.decode());
        document.getElementById('clear-btn').addEventListener('click', () => this.clear());
        document.getElementById('copy-encoded').addEventListener('click', () => this.copyEncoded());
        
        // Auto-encode on input change
        document.getElementById('input-text').addEventListener('input', () => this.autoEncode());
        document.getElementById('position-offset').addEventListener('input', () => this.autoEncode());
        document.getElementById('encoding-variant').addEventListener('change', () => this.autoEncode());
    }
    
    encode() {
        const inputText = document.getElementById('input-text').value;
        const position = parseInt(document.getElementById('position-offset').value) || 0;
        const variant = document.getElementById('encoding-variant').value;
        
        if (!inputText.trim()) {
            this.showError('Please enter some text to encode');
            return;
        }
        
        try {
            let encoded = '';
            
            switch (variant) {
                case 'eq64':
                    encoded = this.encoder.encodeEq64(inputText, position);
                    break;
                case 'shq64':
                    encoded = this.encoder.encodeShq64(inputText, position);
                    break;
                case 't8q64':
                    encoded = this.encoder.encodeT8q64(inputText, position);
                    break;
                case 'zoq64':
                    encoded = this.encoder.encodeZoq64(inputText, position);
                    break;
            }
            
            document.getElementById('encoded-output').value = encoded;
            this.updateAnalysis(inputText, encoded);
            this.updateStepByStep();
            
        } catch (error) {
            this.showError(`Encoding error: ${error.message}`);
        }
    }
    
    decode() {
        const encodedText = document.getElementById('encoded-output').value;
        const position = parseInt(document.getElementById('position-offset').value) || 0;
        
        if (!encodedText.trim()) {
            this.showError('No encoded text to decode');
            return;
        }
        
        try {
            const decoded = this.encoder.decodeEq64(encodedText, position);
            document.getElementById('decoded-output').value = decoded;
        } catch (error) {
            this.showError(`Decoding error: ${error.message}`);
        }
    }
    
    autoEncode() {
        const inputText = document.getElementById('input-text').value;
        if (inputText.trim()) {
            this.encode();
        }
    }
    
    clear() {
        document.getElementById('input-text').value = '';
        document.getElementById('encoded-output').value = '';
        document.getElementById('decoded-output').value = '';
        document.getElementById('position-offset').value = '0';
        document.getElementById('encoding-variant').value = 'eq64';
        
        this.updateAnalysis('', '');
        document.getElementById('step-by-step-output').textContent = 'Click "Encode" to see the detailed encoding process...';
    }
    
    copyEncoded() {
        const encodedText = document.getElementById('encoded-output').value;
        if (encodedText) {
            navigator.clipboard.writeText(encodedText).then(() => {
                const btn = document.getElementById('copy-encoded');
                const originalText = btn.textContent;
                btn.textContent = 'Copied!';
                setTimeout(() => {
                    btn.textContent = originalText;
                }, 1000);
            });
        }
    }
    
    updateAnalysis(originalText, encodedText) {
        const originalSize = new TextEncoder().encode(originalText).length;
        const encodedSize = new TextEncoder().encode(encodedText).length;
        const ratio = originalSize > 0 ? ((encodedSize / originalSize) * 100).toFixed(1) : '0';
        
        document.getElementById('original-size').textContent = `${originalSize} bytes`;
        document.getElementById('encoded-size').textContent = `${encodedSize} bytes`;
        document.getElementById('size-ratio').textContent = `${ratio}%`;
        document.getElementById('position-safety').textContent = '✓ Enabled';
    }
    
    updateStepByStep() {
        const steps = this.encoder.steps.join('\\n');
        document.getElementById('step-by-step-output').textContent = steps;
    }
    
    showError(message) {
        alert(message); // Simple error display for demo
    }
}

// Initialize the demo when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new QuadB64DemoApp();
});
</script>

## Features

This interactive demo showcases:

1. **Real-time Encoding**: See QuadB64 encoding happen as you type
2. **Multiple Variants**: Compare Eq64, Shq64, T8q64, and Zoq64 encodings
3. **Position Context**: Experiment with different position offsets
4. **Step-by-Step Analysis**: Understand the encoding process in detail
5. **Performance Metrics**: See size ratios and encoding efficiency
6. **Copy/Paste Support**: Easy integration with your applications

## How to Use

1. **Enter Text**: Type or paste text in the input area
2. **Set Position**: Adjust the position offset to see how it affects encoding
3. **Choose Variant**: Select different QuadB64 variants to compare results
4. **Encode**: Click "Encode" or let auto-encoding do it for you
5. **Analyze**: Review the step-by-step process and metrics
6. **Copy**: Use the copy button to grab encoded results

## Educational Value

This demo helps you understand:

- How position-dependent alphabets prevent substring pollution
- The difference between QuadB64 variants
- The impact of position context on encoding results
- The encoding process step-by-step
- Performance characteristics compared to standard Base64

Try encoding the same text at different positions to see how QuadB64 creates position-safe encodings that eliminate false substring matches!