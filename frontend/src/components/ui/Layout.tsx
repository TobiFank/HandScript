import { Outlet } from 'react-router-dom';
import Sidebar from '@/components/Sidebar';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import { useState } from 'react';

export default function Layout() {
    const [scrolled, setScrolled] = useState(false);
    const [reachedBottom, setReachedBottom] = useState(false);

    const handleScroll = (event: React.UIEvent<HTMLDivElement>) => {
        const element = event.currentTarget;
        const scrollTop = element.scrollTop;
        const isScrolled = scrollTop > 0;
        const isBottom =
            Math.abs(element.scrollHeight - element.clientHeight - scrollTop) < 1;

        setScrolled(isScrolled);
        setReachedBottom(isBottom);
    };

    return (
        <div className="flex h-screen bg-gray-50">
            <Sidebar />
            <main className="flex-1 flex flex-col overflow-hidden">
                {/* Shadow Overlay */}
                <div
                    className={cn(
                        "absolute top-0 left-0 right-0 h-4 bg-gradient-to-b from-gray-900/10 to-transparent z-10 transition-opacity duration-200",
                        scrolled ? "opacity-100" : "opacity-0"
                    )}
                />

                {/* Main Content Area */}
                <ScrollArea
                    className="flex-1"
                    viewportClassName="min-h-full"
                    onScroll={handleScroll}
                >
                    <Outlet />
                </ScrollArea>

                {/* Bottom Shadow */}
                <div
                    className={cn(
                        "absolute bottom-0 left-0 right-0 h-4 bg-gradient-to-t from-gray-900/10 to-transparent z-10 transition-opacity duration-200",
                        !reachedBottom ? "opacity-100" : "opacity-0"
                    )}
                />
            </main>
        </div>
    );
}