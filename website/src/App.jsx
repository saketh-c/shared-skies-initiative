import { Routes, Route } from 'react-router-dom'
import ScrollToTop from './components/ScrollToTop'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import About from './pages/About'
import Join from './pages/Join'
import MapPage from './pages/MapPage'
import DonatePage from './pages/DonatePage'
import Contact from './pages/Contact'

export default function App() {
  return (
    <>
      <ScrollToTop />
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
        <Route path="/join" element={<Join />} />
        <Route path="/map" element={<MapPage />} />
        <Route path="/donate" element={<DonatePage />} />
        <Route path="/contact" element={<Contact />} />
      </Routes>
    </>
  )
}
