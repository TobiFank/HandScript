// src/App.tsx
import {BrowserRouter as Router, Navigate, Route, Routes} from 'react-router-dom';
import {QueryClient, QueryClientProvider} from '@tanstack/react-query';
import ProjectList from '@/pages/ProjectList';
import ProjectView from '@/pages/ProjectView';
import DocumentView from '@/pages/DocumentView';
import PageEditor from '@/pages/PageEditor';
import DocumentPreview from '@/pages/DocumentPreview';
import WriterManagement from '@/pages/WriterManagement';
import Layout from '@/components/Layout';
import {TooltipProvider} from "@/components/ui/tooltip.tsx";
import WriterDetail from "@/pages/WriterDetail.tsx";

const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            retry: 1,
            refetchOnWindowFocus: false,
        },
    },
});

function App() {
    return (
        <QueryClientProvider client={queryClient}>
            <TooltipProvider>
                <Router>
                    <Routes>
                        <Route element={<Layout/>}>
                            <Route path="/" element={<Navigate to="/projects" replace/>}/>
                            <Route path="/projects" element={<ProjectList/>}/>
                            <Route path="/projects/:projectId" element={<ProjectView/>}/>
                            <Route path="/documents/:documentId" element={<DocumentView/>}/>
                            <Route path="/pages/:pageId/edit" element={<PageEditor/>}/>
                            <Route path="/documents/:documentId/preview" element={<DocumentPreview/>}/>
                            <Route path="/writers" element={<WriterManagement/>}/>
                            <Route path="/writers/:writerId" element={<WriterDetail />} />
                        </Route>
                    </Routes>
                </Router>
            </TooltipProvider>
        </QueryClientProvider>
    );
}

export default App;