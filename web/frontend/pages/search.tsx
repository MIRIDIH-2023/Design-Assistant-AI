import { ChatInput } from "../components/ChatInput";

function Search() {
  return (
    <ChatInput
      sx={{
        width: 700,
        position: "absolute",
        left: "50%",
        bottom: "7%",
        transform: "translateX(-50%)",
      }}
      placeholder="Search for a design templates..."
    />
  );
}

export default Search;
