export const Code = ({ codeBlocks }: { codeBlocks: string[] }) => {
  return (
    <>
      {codeBlocks.map((codeBlock, index) => (
        <pre
          key={index}
          className="bg-muted rounded-md font-medium px-3 py-2 text-xs leading-6"
        >
          {codeBlock}
        </pre>
      ))}
    </>
  );
};
