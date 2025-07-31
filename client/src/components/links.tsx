export const Links = ({ links }: { links: string[] }) => {
  return (
    <ul className="vkui:bg-muted vkui:rounded-md vkui:font-medium vkui:px-3 vkui:py-2 vkui:text-sm vkui:leading-6">
      {links.map((link) => (
        <li key={link} className="vkui:flex vkui:items-center vkui:gap-2">
          <span>🔗</span>
          <a
            href={link}
            target="_blank"
            rel="noopener noreferrer"
            className="vkui:text-agent vkui:hover:underline vkui:hover:text-primary"
          >
            {link}
          </a>
        </li>
      ))}
    </ul>
  );
};
