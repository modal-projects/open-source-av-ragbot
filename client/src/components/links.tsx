export const Links = ({ links }: { links: string[] }) => {
  return (
    <ul className="bg-muted rounded-md font-medium px-3 py-2 text-sm leading-6">
      {links.map((link) => (
        <li key={link} className="flex items-center gap-2">
          <span>🔗</span>
          <a
            href={link}
            target="_blank"
            rel="noopener noreferrer"
            className="text-agent hover:underline hover:text-primary"
          >
            {link}
          </a>
        </li>
      ))}
    </ul>
  );
};
