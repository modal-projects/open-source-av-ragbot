export const Code = ({ codeBlocks }: { codeBlocks: string[] }) => {
  return (
    <>
      {codeBlocks.map((codeBlock, index) => (
        <pre
          key={index}
          className="vkui:bg-muted vkui:rounded-md vkui:font-medium vkui:px-3 vkui:py-2 vkui:text-xs vkui:leading-6"
        >
          {codeBlock}
        </pre>
      ))}
    </>
  );
};
