import { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'

const LINKS = [
  { label: 'About',        to: '/about' },
  { label: 'Join',         to: '/join' },
  { label: 'Real-Time Map',to: '/map' },
  { label: 'Donate',       to: '/donate', highlight: true },
  { label: 'Contact',      to: '/contact' },
]

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const location = useLocation()

  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 50)
    window.addEventListener('scroll', fn, { passive: true })
    return () => window.removeEventListener('scroll', fn)
  }, [])

  useEffect(() => setMenuOpen(false), [location.pathname])

  return (
    <motion.nav
      initial={{ opacity: 0, y: -12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
      style={{
        position: 'fixed', top: 0, left: 0, right: 0, zIndex: 1000,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '0 36px', height: '72px',
        background: scrolled ? 'rgba(4, 10, 36, 0.72)' : 'transparent',
        backdropFilter: scrolled ? 'blur(20px) saturate(180%)' : 'none',
        WebkitBackdropFilter: scrolled ? 'blur(20px) saturate(180%)' : 'none',
        borderBottom: scrolled ? '1px solid rgba(255,255,255,0.07)' : 'none',
        transition: 'background 0.4s, border-color 0.4s',
      }}
    >
      {/* Logo */}
      <Link to="/" style={{
        fontFamily: 'Lora, Georgia, serif',
        fontSize: '17px', fontWeight: 500,
        color: 'rgba(255,255,255,0.95)',
        letterSpacing: '0.01em',
        flexShrink: 0,
      }}>
        Shared Skies Initiative
      </Link>

      {/* Desktop links */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
        {LINKS.map(link => (
          <NavLink
            key={link.to}
            to={link.to}
            active={location.pathname === link.to}
            highlight={link.highlight}
          >
            {link.label}
          </NavLink>
        ))}
      </div>
    </motion.nav>
  )
}

function NavLink({ to, children, active, highlight }) {
  if (highlight) {
    return (
      <Link to={to} style={{
        fontFamily: 'Inter, sans-serif',
        fontSize: '13.5px', fontWeight: 600,
        color: active ? '#0d2478' : '#0d2478',
        background: 'white',
        padding: '7px 18px',
        borderRadius: '20px',
        marginLeft: '8px',
        letterSpacing: '0.005em',
        transition: 'background 0.2s, transform 0.15s',
        display: 'inline-block',
        opacity: active ? 1 : 0.92,
      }}
        onMouseEnter={e => { e.currentTarget.style.background = '#e8f0ff'; e.currentTarget.style.transform = 'scale(1.03)' }}
        onMouseLeave={e => { e.currentTarget.style.background = 'white'; e.currentTarget.style.transform = 'scale(1)' }}
      >
        {children}
      </Link>
    )
  }
  return (
    <Link to={to} style={{
      fontFamily: 'Inter, sans-serif',
      fontSize: '13.5px', fontWeight: 400,
      color: active ? 'rgba(255,255,255,1)' : 'rgba(255,255,255,0.7)',
      padding: '7px 14px',
      borderRadius: '20px',
      letterSpacing: '0.005em',
      transition: 'color 0.2s',
      background: active ? 'rgba(255,255,255,0.1)' : 'transparent',
    }}
      onMouseEnter={e => { e.currentTarget.style.color = 'rgba(255,255,255,1)' }}
      onMouseLeave={e => { e.currentTarget.style.color = active ? 'rgba(255,255,255,1)' : 'rgba(255,255,255,0.7)' }}
    >
      {children}
    </Link>
  )
}
