import { useState } from "react";

function App() {
  const [message, setMessage] = useState("Hello from React (Node-App)!");

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>{message}</h1>
      <p>This is the React SPA running inside Docker.</p>
      <button onClick={() => setMessage("Button clicked! 🚀")}>
        Click Me
      </button>
    </div>
  );
}

export default App;
