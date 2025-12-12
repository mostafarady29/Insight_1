import MainLayout from "@/layouts/MainLayout";
import { useState } from "react";

export default function Contact() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    // For now just open mail client — replace with API when available
    window.location.href = `mailto:officialshoraky@gmail.com?subject=${encodeURIComponent("Contact from " + name)}&body=${encodeURIComponent(message + "\n\nFrom: " + email)}`;
  }

  return (
    <MainLayout>
      <div className="container py-12 max-w-2xl">
        <h1 className="text-3xl font-bold mb-4">Contact</h1>
        <p className="text-muted-foreground mb-6">Have questions or want to partner with us? Send a message and we'll get back to you.</p>

        <form onSubmit={handleSubmit} className="grid gap-4">
          <input value={name} onChange={e=>setName(e.target.value)} placeholder="Your name" className="border border-input px-3 py-2 rounded-none" />
          <input value={email} onChange={e=>setEmail(e.target.value)} placeholder="Your email" className="border border-input px-3 py-2 rounded-none" />
          <textarea value={message} onChange={e=>setMessage(e.target.value)} placeholder="Message" className="border border-input px-3 py-2 rounded-none h-32" />
          <div>
            <button type="submit" className="bg-primary text-primary-foreground px-4 py-2 rounded-none">Send Message</button>
          </div>
        </form>
      </div>
    </MainLayout>
  );
}
