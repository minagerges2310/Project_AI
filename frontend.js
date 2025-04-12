const express = require('express');
const app = express();
const port = 3000;

// Serve static files
app.use(express.static('public'));

// Start the server
app.listen(port, () => {
    console.log(`Node.js server running at http://localhost:${port}`);
});