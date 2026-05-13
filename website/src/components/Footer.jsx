import { motion } from 'framer-motion'

export default function Footer() {
  return (
    <footer style={{ background: '#2d5caa' }}>
      <div style={{ padding: '56px 80px 40px', borderBottom: '1px solid rgba(255,255,255,0.15)' }}>
        <p style={{ fontFamily: 'Inter, sans-serif', fontSize: '11px', fontWeight: 600, color: 'rgba(255,255,255,0.4)', textTransform: 'uppercase', letterSpacing: '0.14em', marginBottom: '16px' }}>
          Socials
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {[
            { label: 'GiveButter', href: 'https://givebutter.com/sharedskies' },
            { label: 'Instagram', href: 'https://www.instagram.com/sharedskiesinitiative' },
          ].map(({ label, href }) => (
            <motion.a
              key={label}
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              whileHover={{ x: 5 }}
              transition={{ duration: 0.2, ease: 'easeOut' }}
              style={{ fontFamily: 'Inter, sans-serif', fontSize: '15px', fontWeight: 400, color: 'rgba(255,255,255,0.8)', display: 'inline-block' }}
            >
              {label}
            </motion.a>
          ))}
        </div>
      </div>
      <div style={{ padding: '24px 80px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontFamily: 'Lora, Georgia, serif', fontSize: '24px', fontWeight: 500, color: 'rgba(255,255,255,0.92)', letterSpacing: '-0.02em' }}>
          Shared Skies
        </span>
        <span style={{ fontFamily: 'Inter, sans-serif', fontSize: '13px', color: 'rgba(255,255,255,0.4)' }}>
          © 2026 Shared Skies Initiative
        </span>
      </div>
    </footer>
  )
}
