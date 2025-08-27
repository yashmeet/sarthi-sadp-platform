/**
 * POML Studio Frontend Integration
 * Real-time prompt engineering with Gemini AI
 */

// API Configuration
const API_BASE_URL = 'https://sadp-healthcare-ai-355881591332.us-central1.run.app';
const WS_URL = 'wss://sadp-healthcare-ai-355881591332.us-central1.run.app';

// Global state
let currentTemplate = null;
let templates = [];
let wsConnection = null;
let testResults = [];

// Initialize POML Studio
async function initializePOMLStudio() {
    console.log('Initializing POML Studio...');
    
    // Load templates
    await loadTemplates();
    
    // Setup WebSocket for live testing
    setupWebSocket();
    
    // Setup event listeners
    setupEventListeners();
    
    // Load default template
    if (templates.length > 0) {
        await loadTemplate(templates[0].id);
    }
}

// Load templates from API
async function loadTemplates() {
    try {
        const response = await fetch(`${API_BASE_URL}/poml/templates`);
        const data = await response.json();
        
        templates = data.templates || [];
        
        // Update template selector
        const selector = document.getElementById('template-select');
        if (selector) {
            selector.innerHTML = '';
            
            // Add default templates if no templates exist
            if (templates.length === 0) {
                templates = getDefaultTemplates();
            }
            
            templates.forEach(template => {
                const option = document.createElement('option');
                option.value = template.id;
                option.textContent = `${template.name} (${template.agent_type})`;
                selector.appendChild(option);
            });
        }
        
        updateTemplateList();
        
    } catch (error) {
        console.error('Failed to load templates:', error);
        // Use default templates as fallback
        templates = getDefaultTemplates();
        updateTemplateList();
    }
}

// Get default templates
function getDefaultTemplates() {
    return [
        {
            id: 'clinical_diagnosis',
            name: 'Clinical Diagnosis',
            agent_type: 'clinical',
            description: 'Diagnostic support with ICD-10 coding',
            content: `<prompt version="2.0" agent="clinical">
  <system>
    You are an expert clinical assistant with 20+ years of experience.
    Follow evidence-based medicine guidelines.
    Prioritize patient safety above all.
  </system>
  
  <context>
    Patient: {{patient_name}}
    Age: {{patient_age}}
    Symptoms: {{symptoms}}
  </context>
  
  <task>
    Analyze the patient data and provide:
    1. Initial assessment
    2. Differential diagnosis (top 3)
    3. Recommended diagnostic tests
    4. Immediate care recommendations
  </task>
  
  <output format="structured">
    Provide response in JSON format with assessment, diagnoses, tests, and recommendations.
  </output>
</prompt>`,
            variables: ['patient_name', 'patient_age', 'symptoms']
        },
        {
            id: 'medication_reconciliation',
            name: 'Medication Reconciliation',
            agent_type: 'medication',
            description: 'Medication history and interaction checking',
            content: `<prompt version="1.0" agent="medication">
  <system>
    You are a clinical pharmacist specializing in medication management.
    Check for drug interactions, contraindications, and dosing issues.
  </system>
  
  <context>
    Patient Medications: {{current_medications}}
    New Prescription: {{new_medication}}
    Medical Conditions: {{conditions}}
  </context>
  
  <task>
    1. Check for drug interactions
    2. Verify appropriate dosing
    3. Identify contraindications
    4. Suggest alternatives if needed
  </task>
</prompt>`,
            variables: ['current_medications', 'new_medication', 'conditions']
        }
    ];
}

// Load a specific template
async function loadTemplate(templateId) {
    try {
        // Try to load from API
        const response = await fetch(`${API_BASE_URL}/poml/templates/${templateId}`);
        
        if (response.ok) {
            currentTemplate = await response.json();
        } else {
            // Use local template
            currentTemplate = templates.find(t => t.id === templateId);
        }
        
        if (currentTemplate) {
            // Update editor
            const editor = document.getElementById('poml-editor');
            if (editor) {
                editor.textContent = currentTemplate.content;
                highlightSyntax();
            }
            
            // Update variables section
            updateVariablesSection();
            
            // Update metrics if available
            if (currentTemplate.executions) {
                updateMetrics(currentTemplate);
            }
        }
        
    } catch (error) {
        console.error('Failed to load template:', error);
        // Use local template as fallback
        currentTemplate = templates.find(t => t.id === templateId);
        if (currentTemplate) {
            document.getElementById('poml-editor').textContent = currentTemplate.content;
        }
    }
}

// Save current template
async function savePrompt() {
    const editor = document.getElementById('poml-editor');
    const content = editor.textContent || editor.innerText;
    
    if (!currentTemplate) {
        // Create new template
        const name = prompt('Enter template name:');
        if (!name) return;
        
        const templateData = {
            name: name,
            agent_type: 'custom',
            description: 'Custom template',
            content: content,
            variables: extractVariables(content),
            tags: []
        };
        
        try {
            const response = await fetch(`${API_BASE_URL}/poml/templates`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(templateData)
            });
            
            if (response.ok) {
                const result = await response.json();
                showNotification('Template saved successfully!', 'success');
                await loadTemplates();
                await loadTemplate(result.template_id);
            } else {
                throw new Error('Failed to save template');
            }
        } catch (error) {
            console.error('Save failed:', error);
            showNotification('Failed to save template. Saved locally.', 'warning');
            
            // Save to localStorage as backup
            localStorage.setItem(`poml_template_${Date.now()}`, JSON.stringify(templateData));
        }
        
    } else {
        // Update existing template
        try {
            const response = await fetch(`${API_BASE_URL}/poml/templates/${currentTemplate.id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    content: content,
                    version: incrementVersion(currentTemplate.version)
                })
            });
            
            if (response.ok) {
                showNotification('Template updated successfully!', 'success');
                await loadTemplate(currentTemplate.id);
            } else {
                throw new Error('Failed to update template');
            }
        } catch (error) {
            console.error('Update failed:', error);
            showNotification('Failed to update template. Changes saved locally.', 'warning');
            localStorage.setItem(`poml_template_${currentTemplate.id}_backup`, content);
        }
    }
}

// Test template with live data
async function testPrompt() {
    const editor = document.getElementById('poml-editor');
    const content = editor.textContent || editor.innerText;
    
    // Get variables
    const variables = {};
    const varInputs = document.querySelectorAll('.variable-input');
    varInputs.forEach(input => {
        variables[input.dataset.variable] = input.value;
    });
    
    // Get test data
    const testData = document.getElementById('test-data-input')?.value || 'Test patient data';
    
    // Show loading state
    showTestLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/poml/test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                template_content: content,
                variables: variables,
                test_data: testData
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            displayTestResult(result);
            
            // Store result
            testResults.push({
                timestamp: new Date(),
                ...result
            });
            
            // Update metrics display
            updateTestMetrics();
        } else {
            throw new Error('Test failed');
        }
        
    } catch (error) {
        console.error('Test failed:', error);
        displayTestError(error.message);
    }
}

// Start A/B test
async function startABTest() {
    // Get template selections
    const templateA = document.getElementById('template-a-select')?.value || currentTemplate?.id;
    const templateB = document.getElementById('template-b-select')?.value;
    
    if (!templateA || !templateB) {
        alert('Please select two templates to compare');
        return;
    }
    
    // Create test cases
    const testCases = [
        {
            variables: { patient_name: 'John Doe', patient_age: '45', symptoms: 'Chest pain, shortness of breath' },
            input: 'Patient presents with acute symptoms'
        },
        {
            variables: { patient_name: 'Jane Smith', patient_age: '62', symptoms: 'Headache, dizziness' },
            input: 'Patient with neurological symptoms'
        }
    ];
    
    showNotification('Starting A/B test...', 'info');
    
    try {
        const response = await fetch(`${API_BASE_URL}/poml/ab-test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: `A/B Test ${new Date().toISOString()}`,
                template_a_id: templateA,
                template_b_id: templateB,
                test_cases: testCases,
                iterations: 5
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            showNotification('A/B test started! Check results in a few moments.', 'success');
            
            // Poll for results
            setTimeout(() => checkABTestResults(result.test_id), 5000);
        } else {
            throw new Error('Failed to start A/B test');
        }
        
    } catch (error) {
        console.error('A/B test failed:', error);
        showNotification('Failed to start A/B test', 'error');
    }
}

// Check A/B test results
async function checkABTestResults(testId) {
    try {
        const response = await fetch(`${API_BASE_URL}/poml/ab-test/${testId}`);
        
        if (response.ok) {
            const results = await response.json();
            displayABTestResults(results);
            
            if (results.status === 'running') {
                // Check again in 5 seconds
                setTimeout(() => checkABTestResults(testId), 5000);
            }
        }
        
    } catch (error) {
        console.error('Failed to get A/B test results:', error);
    }
}

// Setup WebSocket for live testing
function setupWebSocket() {
    try {
        wsConnection = new WebSocket(`${WS_URL}/poml/live-test`);
        
        wsConnection.onopen = () => {
            console.log('WebSocket connected for live testing');
            document.getElementById('ws-status')?.classList.add('connected');
        };
        
        wsConnection.onmessage = (event) => {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        };
        
        wsConnection.onerror = (error) => {
            console.error('WebSocket error:', error);
            document.getElementById('ws-status')?.classList.remove('connected');
        };
        
        wsConnection.onclose = () => {
            console.log('WebSocket disconnected');
            document.getElementById('ws-status')?.classList.remove('connected');
            // Reconnect after 5 seconds
            setTimeout(setupWebSocket, 5000);
        };
        
    } catch (error) {
        console.error('Failed to setup WebSocket:', error);
    }
}

// Handle WebSocket messages
function handleWebSocketMessage(data) {
    switch (data.status) {
        case 'processing':
        case 'executing':
            showTestProgress(data.message);
            break;
        case 'rendered':
            displayRenderedPrompt(data.prompt);
            break;
        case 'complete':
            displayTestResult(data);
            break;
        case 'error':
            displayTestError(data.message);
            break;
    }
}

// Live test via WebSocket
function liveTest() {
    if (!wsConnection || wsConnection.readyState !== WebSocket.OPEN) {
        showNotification('WebSocket not connected. Using HTTP instead.', 'warning');
        testPrompt();
        return;
    }
    
    const editor = document.getElementById('poml-editor');
    const content = editor.textContent || editor.innerText;
    
    const variables = {};
    const varInputs = document.querySelectorAll('.variable-input');
    varInputs.forEach(input => {
        variables[input.dataset.variable] = input.value;
    });
    
    const testData = document.getElementById('test-data-input')?.value || 'Test data';
    
    wsConnection.send(JSON.stringify({
        action: 'test',
        template_content: content,
        variables: variables,
        test_data: testData
    }));
    
    showTestLoading();
}

// UI Helper Functions
function updateVariablesSection() {
    const variables = extractVariables(currentTemplate?.content || '');
    const container = document.getElementById('variables-section');
    
    if (!container) return;
    
    container.innerHTML = '<h4 class="font-semibold mb-2">Template Variables</h4>';
    
    variables.forEach(variable => {
        const div = document.createElement('div');
        div.className = 'mb-2';
        div.innerHTML = `
            <label class="block text-sm font-medium text-gray-700">${variable}</label>
            <input type="text" 
                   class="variable-input mt-1 block w-full rounded-md border-gray-300 shadow-sm" 
                   data-variable="${variable}"
                   placeholder="Enter ${variable}">
        `;
        container.appendChild(div);
    });
}

function extractVariables(content) {
    const matches = content.match(/\{\{(\w+)\}\}/g) || [];
    return [...new Set(matches.map(m => m.replace(/[{}]/g, '')))];
}

function updateMetrics(template) {
    document.getElementById('metric-executions').textContent = template.executions || 0;
    document.getElementById('metric-success-rate').textContent = 
        `${((template.success_rate || 0) * 100).toFixed(1)}%`;
    document.getElementById('metric-latency').textContent = 
        `${(template.avg_latency || 0).toFixed(0)}ms`;
}

function updateTestMetrics() {
    if (testResults.length === 0) return;
    
    const successCount = testResults.filter(r => r.success).length;
    const avgLatency = testResults.reduce((sum, r) => sum + r.latency_ms, 0) / testResults.length;
    const avgTokens = testResults.reduce((sum, r) => sum + r.tokens_used, 0) / testResults.length;
    
    document.getElementById('test-count').textContent = testResults.length;
    document.getElementById('test-success-rate').textContent = 
        `${((successCount / testResults.length) * 100).toFixed(1)}%`;
    document.getElementById('test-avg-latency').textContent = `${avgLatency.toFixed(0)}ms`;
    document.getElementById('test-avg-tokens').textContent = avgTokens.toFixed(0);
}

function displayTestResult(result) {
    const container = document.getElementById('test-results');
    if (!container) return;
    
    const resultDiv = document.createElement('div');
    resultDiv.className = `p-4 mb-4 rounded-lg ${result.success ? 'bg-green-50' : 'bg-red-50'}`;
    resultDiv.innerHTML = `
        <div class="flex justify-between items-start mb-2">
            <span class="font-semibold ${result.success ? 'text-green-800' : 'text-red-800'}">
                ${result.success ? '✓ Success' : '✗ Failed'}
            </span>
            <span class="text-sm text-gray-600">
                ${result.latency_ms.toFixed(0)}ms | ${result.tokens_used} tokens
            </span>
        </div>
        <div class="bg-white p-3 rounded border">
            <pre class="whitespace-pre-wrap text-sm">${escapeHtml(result.result)}</pre>
        </div>
    `;
    
    container.insertBefore(resultDiv, container.firstChild);
    
    // Keep only last 5 results
    while (container.children.length > 5) {
        container.removeChild(container.lastChild);
    }
}

function displayABTestResults(results) {
    const container = document.getElementById('ab-test-results');
    if (!container) return;
    
    const winner = results.winner === 'template_a' ? 'Template A' : 
                   results.winner === 'template_b' ? 'Template B' : 
                   'No clear winner';
    
    container.innerHTML = `
        <div class="bg-white p-6 rounded-lg shadow">
            <h3 class="text-xl font-semibold mb-4">A/B Test Results</h3>
            <div class="grid grid-cols-2 gap-4 mb-4">
                <div class="p-4 bg-blue-50 rounded">
                    <h4 class="font-semibold text-blue-800">Template A</h4>
                    <p>Success Rate: ${(results.template_a.success_rate * 100).toFixed(1)}%</p>
                    <p>Avg Latency: ${results.template_a.avg_latency.toFixed(0)}ms</p>
                    <p>Executions: ${results.template_a.executions}</p>
                </div>
                <div class="p-4 bg-green-50 rounded">
                    <h4 class="font-semibold text-green-800">Template B</h4>
                    <p>Success Rate: ${(results.template_b.success_rate * 100).toFixed(1)}%</p>
                    <p>Avg Latency: ${results.template_b.avg_latency.toFixed(0)}ms</p>
                    <p>Executions: ${results.template_b.executions}</p>
                </div>
            </div>
            <div class="text-center p-4 bg-gray-100 rounded">
                <p class="text-lg font-semibold">Winner: ${winner}</p>
                ${results.confidence ? `<p class="text-sm text-gray-600">Confidence: ${(results.confidence * 100).toFixed(1)}%</p>` : ''}
            </div>
        </div>
    `;
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 ${
        type === 'success' ? 'bg-green-500' :
        type === 'error' ? 'bg-red-500' :
        type === 'warning' ? 'bg-yellow-500' :
        'bg-blue-500'
    } text-white`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

function showTestLoading() {
    const container = document.getElementById('test-results');
    if (!container) return;
    
    const loader = document.createElement('div');
    loader.id = 'test-loader';
    loader.className = 'p-4 mb-4 bg-gray-100 rounded-lg text-center';
    loader.innerHTML = `
        <div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
        <p class="mt-2 text-sm text-gray-600">Testing template...</p>
    `;
    
    container.insertBefore(loader, container.firstChild);
}

function showTestProgress(message) {
    const loader = document.getElementById('test-loader');
    if (loader) {
        const p = loader.querySelector('p');
        if (p) p.textContent = message;
    }
}

function displayTestError(error) {
    const loader = document.getElementById('test-loader');
    if (loader) loader.remove();
    
    displayTestResult({
        success: false,
        result: error,
        latency_ms: 0,
        tokens_used: 0
    });
}

function highlightSyntax() {
    const editor = document.getElementById('poml-editor');
    if (!editor) return;
    
    let content = editor.textContent || editor.innerText;
    
    // Simple syntax highlighting
    content = content.replace(/(<[^>]+>)/g, '<span class="text-blue-600">$1</span>');
    content = content.replace(/(\{\{[^}]+\}\})/g, '<span class="text-green-600 font-bold">$1</span>');
    
    editor.innerHTML = content;
}

function updateTemplateList() {
    const list = document.getElementById('template-list');
    if (!list) return;
    
    list.innerHTML = '';
    
    templates.forEach(template => {
        const item = document.createElement('div');
        item.className = 'p-3 border rounded hover:bg-gray-50 cursor-pointer';
        item.innerHTML = `
            <div class="font-semibold">${template.name}</div>
            <div class="text-sm text-gray-600">${template.description}</div>
            <div class="text-xs text-gray-500 mt-1">
                Type: ${template.agent_type} | 
                ${template.executions || 0} executions
            </div>
        `;
        item.onclick = () => loadTemplate(template.id);
        list.appendChild(item);
    });
}

function incrementVersion(version) {
    const parts = version.split('.');
    const patch = parseInt(parts[parts.length - 1] || 0) + 1;
    parts[parts.length - 1] = patch;
    return parts.join('.');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function setupEventListeners() {
    // Template selector
    const selector = document.getElementById('template-select');
    if (selector) {
        selector.addEventListener('change', (e) => loadTemplate(e.target.value));
    }
    
    // Editor changes
    const editor = document.getElementById('poml-editor');
    if (editor) {
        editor.addEventListener('input', () => {
            highlightSyntax();
            updateVariablesSection();
        });
    }
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            switch (e.key) {
                case 's':
                    e.preventDefault();
                    savePrompt();
                    break;
                case 'Enter':
                    e.preventDefault();
                    testPrompt();
                    break;
            }
        }
    });
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializePOMLStudio);
} else {
    initializePOMLStudio();
}

// Export functions for use in HTML
window.savePrompt = savePrompt;
window.testPrompt = testPrompt;
window.startABTest = startABTest;
window.liveTest = liveTest;
window.loadTemplate = loadTemplate;